import requests
import numpy as np
import jmespath
from typing import List, Dict, Any, Optional, Set, Tuple
import time
from tqdm import tqdm
import json
import os
import gzip


class EMDBRandomSampler:
    """
    A client for randomly sampling entries from the EMDB (Electron Microscopy Data Bank) API.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility, by default 42
    min_id : int, optional
        Minimum EMDB ID to sample from, by default 20_000
    max_id : int, optional
        Maximum EMDB ID to sample from, by default 50_000

    Attributes
    ----------
    BASE_URL : str
        Base URL for the EMDB API
    rng : np.random.Generator
        NumPy random number generator
    min_id : int
        Minimum EMDB ID to sample from
    max_id : int
        Maximum EMDB ID to sample from
    session : requests.Session
        HTTP session for making API requests
    """

    BASE_URL = "https://www.ebi.ac.uk/emdb/api"

    def __init__(self, seed: int = 42, min_id: int = 20_000, max_id: int = 50_000):
        """
        Initialize the EMDB random sampler with specified parameters.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42
        min_id : int, optional
            Minimum EMDB ID to sample from, by default 20_000
        max_id : int, optional
            Maximum EMDB ID to sample from, by default 50_000
        """
        self.rng = np.random.default_rng(seed)
        self.min_id = min_id
        self.max_id = max_id
        self.session = requests.Session()

    def _format_emdb_id(self, emdb_id: int) -> str:
        """
        Format an integer ID to standard EMDB ID string format.

        Parameters
        ----------
        emdb_id : int
            Numeric EMDB ID

        Returns
        -------
        str
            Formatted EMDB ID (e.g., "1001")
        """
        return str(emdb_id)

    def _extract_method(self, entry_info: Dict[str, Any]) -> Optional[str]:
        """
        Extract the method from an entry's information.

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        Optional[str]
            Method used, or None if not found or invalid
        """
        # Query the method directly using JMESPath
        return jmespath.search(
            "structure_determination_list.structure_determination[0].method", entry_info
        )

    def _extract_resolution(self, entry_info: Dict[str, Any]) -> Optional[float]:
        """
        Extract the resolution from an entry's information.

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        Optional[float]
            Resolution in Angstroms, or None if not found or invalid
        """
        # JMESPath query to find any resolution value in the nested structure
        # This flattens all image_processing items across all structure_determinations
        query = """
        structure_determination_list.structure_determination[*].
        image_processing[*].
        final_reconstruction.resolution.valueOf_
        """

        # This will find all resolution values and flatten them into a single array
        resolutions = jmespath.search(query, entry_info)

        # If we found any resolutions, try to convert the first valid one to float
        if resolutions:
            try:
                return float(resolutions[0][0])
            except (ValueError, TypeError):
                pass

        return None

    def _check_cubic_dimensions(self, entry_info: Dict[str, Any]) -> bool:
        """
        Check if an entry has cubic dimensions and size is not too large.

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        bool
            True if dimensions are cubic and not too large, False otherwise
        """
        # Extract dimensions using JMESPath
        col = jmespath.search("map.dimensions.col", entry_info)
        row = jmespath.search("map.dimensions.row", entry_info)
        sec = jmespath.search("map.dimensions.sec", entry_info)

        # All dimensions must exist and be positive
        if not (col and row and sec):
            return False

        try:
            col = int(col)
            row = int(row)
            sec = int(sec)

            if col <= 0 or row <= 0 or sec <= 0:
                return False

            # Check if dimensions are exactly cubic
            if col != row or row != sec or col != sec:
                return False

            # Check if dimensions are not too large (max 512)
            if (
                col > 512
            ):  # Since we know col = row = sec, we only need to check one dimension
                return False

            return True

        except (ValueError, TypeError):
            return False

    def _check_pixel_spacing(self, entry_info: Dict[str, Any]) -> bool:
        """
        Check if pixel spacing is smaller than 10 Angstroms.

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        bool
            True if pixel spacing is valid, False otherwise
        """
        # Extract pixel spacing using JMESPath
        x_spacing = jmespath.search("map.pixel_spacing.x.valueOf_", entry_info)
        y_spacing = jmespath.search("map.pixel_spacing.y.valueOf_", entry_info)
        z_spacing = jmespath.search("map.pixel_spacing.z.valueOf_", entry_info)

        # All spacing values must exist
        if not (x_spacing and y_spacing and z_spacing):
            return False

        try:
            x_spacing = float(x_spacing)
            y_spacing = float(y_spacing)
            z_spacing = float(z_spacing)

            # All spacing values must be smaller than 10 Angstroms
            if x_spacing >= 10.0 or y_spacing >= 10.0 or z_spacing >= 10.0:
                return False

            return True

        except (ValueError, TypeError):
            return False

    def _check_half_maps(self, entry_info: Dict[str, Any]) -> bool:
        """
        Check if entry has at least two half maps.

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        bool
            True if at least two half maps are present, False otherwise
        """
        # Use JMESPath to count the half maps
        half_maps = jmespath.search("interpretation.half_map_list.half_map", entry_info)

        # Must have at least two half maps
        if not half_maps or not isinstance(half_maps, list) or len(half_maps) < 2:
            return False

        return True

    def check_entry_quality(
        self, entry_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if an EMDB entry meets all quality criteria and extract key metadata:
        - Has valid method (singleParticle, subtomogramAveraging, or helical)
        - Has valid resolution data (specified and < 10 Å)
        - Has cubic dimensions and not too large (≤ 512)
        - Has pixel spacing smaller than 10 Angstroms
        - Has two half maps present

        Parameters
        ----------
        entry_info : Dict[str, Any]
            Dictionary containing entry information

        Returns
        -------
        Tuple[bool, Optional[str], Optional[float]]
            A tuple containing:
            - Boolean indicating if entry meets quality criteria
            - Method string if valid, None otherwise
            - Resolution float if valid, None otherwise
        """
        # Check if method exists and is one of the accepted methods
        allowed_methods = ["singleParticle", "subtomogramAveraging", "helical"]
        method = self._extract_method(entry_info)

        if method not in allowed_methods:
            return False, None, None

        # Check if resolution exists and is better than 10 Å (smaller is better)
        resolution = self._extract_resolution(entry_info)
        if resolution is None or resolution >= 10.0:
            return False, method, resolution

        # Check for cubic dimensions
        if not self._check_cubic_dimensions(entry_info):
            return False, method, resolution

        # Check for valid pixel spacing (< 10 Å)
        if not self._check_pixel_spacing(entry_info):
            return False, method, resolution

        # Check for presence of two half maps
        if not self._check_half_maps(entry_info):
            return False, method, resolution

        return True, method, resolution

    def get_entry_info(self, emdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about an EMDB entry.

        Parameters
        ----------
        emdb_id : int
            The EMDB ID to retrieve information for

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing entry information, or None if the entry doesn't exist
        """
        formatted_id = self._format_emdb_id(emdb_id)
        url = f"{self.BASE_URL}/entry/{formatted_id}"

        try:
            response = self.session.get(url)
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    print(f"Failed to parse JSON for EMD-{formatted_id}")
                    return None
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving EMDB entry EMD-{formatted_id}: {e}")
            return None

    def get_sample_entries(
        self, n_samples: int = 110, retry_limit: int = 1000, delay: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Sample random valid EMDB entries.

        Parameters
        ----------
        n_samples : int, optional
            Number of valid entries to sample, by default 110
        retry_limit : int, optional
            Maximum number of attempts to find valid entries, by default 1000
        delay : float, optional
            Delay between API requests in seconds, by default 0.2

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing entry information with added metadata

        Raises
        ------
        RuntimeError
            If unable to find the requested number of valid entries within the retry limit
        """
        valid_entries = []
        tried_ids: Set[int] = set()
        attempts = 0

        print(
            f"Sampling {n_samples} random EMDB entries between IDs {self.min_id} and {self.max_id}"
        )

        while len(valid_entries) < n_samples and attempts < retry_limit:
            # Generate a random ID that hasn't been tried yet
            while True:
                random_id = self.rng.integers(self.min_id, self.max_id + 1)
                if random_id not in tried_ids:
                    tried_ids.add(random_id)
                    break

            attempts += 1

            # Check if the entry meets our quality criteria
            if attempts % 10 == 0:
                print(
                    f"Progress: Found {len(valid_entries)}/{n_samples} entries after {attempts} attempts"
                )

            entry_info = self.get_entry_info(random_id)
            if entry_info:
                # Check quality and extract metadata in one step
                is_valid, method, resolution = self.check_entry_quality(entry_info)

                if is_valid:
                    # Add metadata to entry_info to avoid re-extracting later
                    entry_info["_metadata"] = {
                        "method": method,
                        "resolution": resolution,
                        "is_valid": True,
                    }

                    valid_entries.append(entry_info)
                    print(
                        f"Found valid entry EMD-{random_id} ({len(valid_entries)}/{n_samples})"
                    )

                    # Extract information using JMESPath
                    title = jmespath.search("admin.title", entry_info) or "N/A"

                    # Use the already extracted metadata
                    resolution_display = (
                        f"{resolution} Å" if resolution is not None else "N/A"
                    )
                    method_display = method if method is not None else "N/A"

                    print(f"  Title: {title}")
                    print(f"  Resolution: {resolution_display}")
                    print(f"  Method: {method_display}")

                    # Extract dimensions with JMESPath
                    col = jmespath.search("map.dimensions.col", entry_info) or "N/A"
                    row = jmespath.search("map.dimensions.row", entry_info) or "N/A"
                    sec = jmespath.search("map.dimensions.sec", entry_info) or "N/A"

                    if col != "N/A" and row != "N/A" and sec != "N/A":
                        print(f"  Dimensions: {col} × {row} × {sec}")

                        # Show pixel spacing if available with JMESPath
                        x_spacing = (
                            jmespath.search("map.pixel_spacing.x.valueOf_", entry_info)
                            or "N/A"
                        )
                        y_spacing = (
                            jmespath.search("map.pixel_spacing.y.valueOf_", entry_info)
                            or "N/A"
                        )
                        z_spacing = (
                            jmespath.search("map.pixel_spacing.z.valueOf_", entry_info)
                            or "N/A"
                        )

                        if (
                            x_spacing != "N/A"
                            and y_spacing != "N/A"
                            and z_spacing != "N/A"
                        ):
                            print(
                                f"  Pixel spacing: {x_spacing} × {y_spacing} × {z_spacing} Å"
                            )

                    print(f"  EMDB ID: EMD-{random_id}")

            # Add a small delay to avoid overloading the API
            time.sleep(delay)

        if len(valid_entries) < n_samples:
            print(
                f"Warning: Only found {len(valid_entries)} valid entries after {attempts} attempts"
            )

        return valid_entries

    def download_half_maps(
        self,
        entries: List[Dict[str, Any]],
        output_dir: str = "emdb_half_maps",
        decompress: bool = True,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
    ) -> Dict[str, List[str]]:
        """
        Download half map files for the given entries and optionally decompress them.
        Uses exponential backoff retry logic to handle connection issues.

        Parameters
        ----------
        entries : List[Dict[str, Any]]
            List of entry dictionaries
        output_dir : str, optional
            Directory to save map files to, by default "emdb_half_maps"
        decompress : bool, optional
            Whether to decompress downloaded .gz files, by default True
        max_retries : int, optional
            Maximum number of retry attempts per file, by default 5
        initial_delay : float, optional
            Initial delay between retries in seconds, by default 2.0
        max_delay : float, optional
            Maximum delay between retries in seconds, by default 60.0

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping EMDB IDs to lists of paths to downloaded files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        downloaded_files = {}

        for entry_index, entry in enumerate(entries):
            # Add delay between entries to avoid overwhelming the server
            if entry_index > 0:
                time.sleep(2.0)  # 2 second delay between entries

            emdb_id = entry.get("emdb_id", "").replace("EMD-", "")
            if not emdb_id:
                continue

            entry_downloads = []

            # Get half map filenames from entry info using JMESPath
            half_map_files = jmespath.search(
                "interpretation.half_map_list.half_map[].file", entry
            )

            if (
                not half_map_files
                or not isinstance(half_map_files, list)
                or len(half_map_files) < 2
            ):
                print(f"Warning: No half maps found for EMD-{emdb_id}")
                continue

            for i, half_map_file in enumerate(half_map_files, 1):
                if not half_map_file:
                    continue

                # Add delay between half maps within the same entry
                if i > 1:
                    time.sleep(1.0)  # 1 second delay between half maps

                # Half maps are in the "other" directory (not in "map")
                half_map_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/other/{half_map_file}"
                compressed_path = os.path.join(
                    output_dir, f"emd_{emdb_id}_half_map_{i}.map.gz"
                )
                decompressed_path = os.path.join(
                    output_dir, f"emd_{emdb_id}_half_map_{i}.map"
                )

                # Skip download if file already exists
                if decompress and os.path.exists(decompressed_path):
                    print(f"File already exists: {decompressed_path}")
                    entry_downloads.append(decompressed_path)
                    continue
                elif not decompress and os.path.exists(compressed_path):
                    print(f"File already exists: {compressed_path}")
                    entry_downloads.append(compressed_path)
                    continue

                print(f"Downloading half map {i} for EMD-{emdb_id}...")

                # Implement retry logic with exponential backoff
                retry_count = 0
                delay = initial_delay
                download_success = False

                while retry_count < max_retries and not download_success:
                    try:
                        # First make a HEAD request to get the content length
                        head_response = self.session.head(half_map_url, timeout=30)
                        head_response.raise_for_status()
                        total_size = int(head_response.headers.get("content-length", 0))

                        # Now stream the download with progress tracking
                        response = self.session.get(
                            half_map_url, stream=True, timeout=30
                        )
                        response.raise_for_status()

                        # Set up progress bar
                        progress_bar = tqdm(
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            desc=f"EMD-{emdb_id} half map {i}",
                        )

                        with open(compressed_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    progress_bar.update(len(chunk))

                        progress_bar.close()
                        print(f"Downloaded {compressed_path}")
                        download_success = True

                    except (
                        requests.exceptions.RequestException,
                        ConnectionError,
                        TimeoutError,
                    ) as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Connection error: {e}")
                            print(
                                f"Retrying in {delay:.1f} seconds... (Attempt {retry_count}/{max_retries})"
                            )
                            time.sleep(delay)
                            # Exponential backoff with jitter
                            delay = min(
                                delay * 2 * (1 + 0.1 * np.random.random()), max_delay
                            )
                        else:
                            print(
                                f"Failed to download after {max_retries} attempts: {e}"
                            )
                            break

                if not download_success:
                    continue

                # Decompress if requested and file ends with .gz
                if (
                    decompress
                    and compressed_path.endswith(".gz")
                    and os.path.exists(compressed_path)
                ):
                    print(f"Decompressing {compressed_path}...")
                    try:
                        # Get the compressed file size for progress tracking
                        compressed_size = os.path.getsize(compressed_path)

                        # Set up decompression progress bar
                        decomp_progress = tqdm(
                            total=compressed_size,
                            unit="iB",
                            unit_scale=True,
                            desc=f"Decompressing half map {i}",
                        )

                        # Custom chunk-by-chunk decompression with progress tracking
                        with gzip.open(compressed_path, "rb") as f_in:
                            with open(decompressed_path, "wb") as f_out:
                                while True:
                                    chunk = f_in.read(8192)
                                    if not chunk:
                                        break
                                    f_out.write(chunk)
                                    # Approximating progress based on input consumed
                                    decomp_progress.update(len(chunk))

                        decomp_progress.close()
                        print(f"Decompressed to {decompressed_path}")
                        entry_downloads.append(decompressed_path)

                        # Optionally remove the compressed file after decompression
                        os.remove(compressed_path)
                        print(f"Removed compressed file {compressed_path}")
                    except Exception as e:
                        print(f"Error decompressing {compressed_path}: {e}")
                        entry_downloads.append(
                            compressed_path
                        )  # Keep the compressed file if decompression fails
                else:
                    entry_downloads.append(compressed_path)

            if entry_downloads:
                downloaded_files[f"EMD-{emdb_id}"] = entry_downloads

        return downloaded_files


if __name__ == "__main__":
    # # Create a sampler with a specific seed for reproducibility
    sampler = EMDBRandomSampler(seed=42, min_id=20_000, max_id=50_000)
    #
    # # Sample random EMDB entries
    # print("Starting search for high-quality EMDB entries...")
    # print("Criteria:")
    # print("  - Method must be singleParticle, subtomogramAveraging, or helical")
    # print("  - Resolution must be specified and better than 10 Å")
    # print("  - Dimensions must be exactly cubic")
    # print("  - Dimensions must not exceed 512")
    # print("  - Pixel spacing must be smaller than 10 Angstroms")
    # print("  - Must have at least two half maps")
    # print()
    #
    # random_entries = sampler.get_sample_entries(n_samples=110)
    #
    # print(f"\nSuccessfully found {len(random_entries)} high-quality EMDB entries")
    #
    # # Use JMESPath to extract and analyze metadata from all entries
    # # Create a JMESPath expression to extract all methods and resolutions
    # methods_query = "[*]._metadata.method"
    # resolutions_query = "[*]._metadata.resolution"
    #
    # # Extract all methods and resolutions in one go
    # all_methods = jmespath.search(methods_query, random_entries)
    # all_resolutions = [r for r in jmespath.search(resolutions_query, random_entries) if
    #                    r is not None]
    #
    # # Count method frequency
    # methods = {}
    # for method in all_methods:
    #     if method:
    #         methods[method] = methods.get(method, 0) + 1
    #
    # # Print statistics
    # print("\nMethod distribution:")
    # for method, count in methods.items():
    #     print(
    #         f"  - {method}: {count} entries ({count / len(random_entries) * 100:.1f}%)")
    #
    # if all_resolutions:
    #     avg_resolution = sum(all_resolutions) / len(all_resolutions)
    #     min_resolution = min(all_resolutions)
    #     max_resolution = max(all_resolutions)
    #     print(f"\nResolution statistics:")
    #     print(f"  - Average: {avg_resolution:.2f} Å")
    #     print(f"  - Range: {min_resolution:.2f} - {max_resolution:.2f} Å")
    #
    # # Example: Save the entries to a file
    # with open("random_emdb_entries.json", "w") as f:
    #     json.dump(random_entries, f, indent=2)
    #
    # print("\nSaved entries to random_emdb_entries.json")

    with open("random_emdb_entries.json") as f:
        random_entries = json.load(f)

    # Download and decompress half maps
    print("\nDownloading half maps...")
    downloaded_maps = sampler.download_half_maps(random_entries, decompress=True)

    # Count total downloaded files
    total_files = sum(len(files) for files in downloaded_maps.values())
    print(
        f"\nDownloaded and decompressed {total_files} half map files for {len(downloaded_maps)} entries"
    )

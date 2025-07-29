import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from miss_alignment.models.models import MissAlignment, TripletMarginRankingLoss


class TestMissAlignment:
    """
    Tests for the MissAlignment class.
    """

    @pytest.fixture
    def model(self):
        """
        Fixture that returns a default MissAlignment model.

        Returns
        -------
        MissAlignment
            A MissAlignment model with default parameters.
        """
        return MissAlignment()

    @pytest.fixture
    def custom_model(self):
        """
        Fixture that returns a MissAlignment model with custom parameters.

        Returns
        -------
        MissAlignment
            A MissAlignment model with custom parameters.
        """
        return MissAlignment(
            learning_rate=1e-3,
            warmup_steps=100,
            weight_decay=0.01,
            margin=0.3,
            lr_scheduler={"mode": "min", "factor": 0.1, "patience": 5},
        )

    @pytest.fixture
    def sample_batch(self):
        """
        Fixture that returns a sample batch for testing.

        Returns
        -------
        tuple
            A tuple containing (img1, img2, img3, target) where:
            - img1, img2, img3 are tensors of shape (2, 1, 64, 64, 64)
            - target is a tensor of shape (2, 3) containing triplet indices
        """
        batch_size = 2
        img1 = torch.randn(batch_size, 1, 64, 64, 64)
        img2 = torch.randn(batch_size, 1, 64, 64, 64)
        img3 = torch.randn(batch_size, 1, 64, 64, 64)
        
        # Create target tensor with shape (batch_size, 3)
        # First example: [1, 1, -1] (img1 and img2 are aligned, img3 is misaligned)
        # Second example: [-1, -1, 1] (img1 and img2 are misaligned, img3 is aligned)
        target = torch.tensor([[1, 1, -1], [-1, -1, 1]])
        
        return img1, img2, img3, target

    def test_init_default(self, model):
        """
        Test initialization with default parameters.

        Parameters
        ----------
        model : MissAlignment
            The default model fixture.
        """
        assert model.learning_rate == 1e-4
        assert model.warmup_steps == 500
        assert model.weight_decay == 0
        assert isinstance(model.criterion, TripletMarginRankingLoss)
        assert model.criterion.margin == 0.5
        assert model.hparams.learning_rate == 1e-4
        assert model.hparams.warmup_steps == 500
        assert model.hparams.weight_decay == 0
        assert model.hparams.margin == 0.5
        assert model.hparams.lr_scheduler is None

    def test_init_custom(self, custom_model):
        """
        Test initialization with custom parameters.

        Parameters
        ----------
        custom_model : MissAlignment
            The custom model fixture.
        """
        assert custom_model.learning_rate == 1e-3
        assert custom_model.warmup_steps == 100
        assert custom_model.weight_decay == 0.01
        assert custom_model.criterion.margin == 0.3
        assert custom_model.hparams.learning_rate == 1e-3
        assert custom_model.hparams.warmup_steps == 100
        assert custom_model.hparams.weight_decay == 0.01
        assert custom_model.hparams.margin == 0.3
        assert custom_model.hparams.lr_scheduler == {"mode": "min", "factor": 0.1, "patience": 5}

    def test_forward(self, model):
        """
        Test the forward pass of the model.

        Parameters
        ----------
        model : MissAlignment
            The default model fixture.
        """
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 64, 64)
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        # Check output type
        assert isinstance(output, torch.Tensor)

    def test_common_step(self, model, sample_batch):
        """
        Test the _common_step method.

        Parameters
        ----------
        model : MissAlignment
            The default model fixture.
        sample_batch : tuple
            The sample batch fixture.
        """
        loss, batch_size, score_aligned, score_misaligned = model._common_step(sample_batch)
        
        # Check types and shapes
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Scalar tensor
        assert batch_size == 2
        assert isinstance(score_aligned, torch.Tensor)
        assert isinstance(score_misaligned, torch.Tensor)
        assert score_aligned.shape == torch.Size([])  # Scalar tensor
        assert score_misaligned.shape == torch.Size([])  # Scalar tensor

    def test_training_step(self, model, sample_batch, monkeypatch):
        """
        Test the training_step method.

        Parameters
        ----------
        model : MissAlignment
            The default model fixture.
        sample_batch : tuple
            The sample batch fixture.
        monkeypatch : pytest.MonkeyPatch
            Pytest's monkeypatch fixture for mocking.
        """
        # Mock the log method to avoid errors during testing
        logged_values = {}
        def mock_log(name, value, batch_size, prog_bar, on_step):
            logged_values[name] = value
        
        monkeypatch.setattr(model, "log", mock_log)
        
        # Mock the optimizers method
        class MockOptimizer:
            param_groups = [{"lr": 1e-4}]
        
        def mock_optimizers():
            return MockOptimizer()
        
        monkeypatch.setattr(model, "optimizers", mock_optimizers)
        
        # Call training_step
        loss = model.training_step(sample_batch, 0)
        
        # Check that loss is returned and logged values are correct
        assert isinstance(loss, torch.Tensor)
        assert "train loss" in logged_values
        assert "train score aligned" in logged_values
        assert "train score misaligned" in logged_values
        assert "learning rate" in logged_values
        assert logged_values["learning rate"] == 1e-4

    def test_configure_optimizers_no_weight_decay(self, model):
        """
        Test configure_optimizers with no weight decay.

        Parameters
        ----------
        model : MissAlignment
            The default model fixture.
        """
        optimizer_config = model.configure_optimizers()
        
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.Adam)
        assert not isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
        assert optimizer_config["lr_scheduler"]["interval"] == "step"

    def test_configure_optimizers_with_weight_decay(self):
        """
        Test configure_optimizers with weight decay.
        """
        model = MissAlignment(weight_decay=0.01)
        optimizer_config = model.configure_optimizers()
        
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
        assert optimizer_config["optimizer"].defaults["weight_decay"] == 0.01

    def test_configure_optimizers_with_lr_scheduler(self, custom_model):
        """
        Test configure_optimizers with lr_scheduler.

        Parameters
        ----------
        custom_model : MissAlignment
            The custom model fixture with lr_scheduler.
        """
        optimizer_config = custom_model.configure_optimizers()
        
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert isinstance(optimizer_config["lr_scheduler"], list)
        assert len(optimizer_config["lr_scheduler"]) == 2
        
        # Check warmup scheduler
        assert optimizer_config["lr_scheduler"][0]["interval"] == "step"
        
        # Check plateau scheduler
        assert optimizer_config["lr_scheduler"][1]["interval"] == "epoch"
        assert optimizer_config["lr_scheduler"][1]["monitor"] == "train loss"
        
        plateau_scheduler = optimizer_config["lr_scheduler"][1]["scheduler"]
        assert plateau_scheduler.factor == 0.1
        assert plateau_scheduler.patience == 5
        assert plateau_scheduler.mode == "min"


class TestTripletMarginRankingLoss:
    """
    Tests for the TripletMarginRankingLoss class.
    """

    def test_init_default(self):
        """
        Test initialization with default parameters.
        """
        loss_fn = TripletMarginRankingLoss()
        assert loss_fn.margin == 1.0
        assert loss_fn.reduction == "mean"

    def test_init_custom(self):
        """
        Test initialization with custom parameters.
        """
        loss_fn = TripletMarginRankingLoss(margin=0.5, reduction="sum")
        assert loss_fn.margin == 0.5
        assert loss_fn.reduction == "sum"

    def test_forward(self):
        """
        Test the forward pass of the loss function.
        """
        loss_fn = TripletMarginRankingLoss(margin=0.5)
        
        # Create embeddings tensor with shape (2, 3)
        embeddings = torch.tensor([
            [1.0, 2.0, -4.0],
            [-4.0, -3.0, -2.0],
        ], dtype=torch.float32)
        
        # Create triplet indices tensor with shape (2, 3)
        triplet_indices = torch.tensor([
            [1, 1, -1],
            [1, 1, -1]
        ])
        
        loss = loss_fn(embeddings, triplet_indices)
        
        # Check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # Loss should be non-negative
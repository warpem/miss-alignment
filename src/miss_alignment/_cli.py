from click import Context
import typer
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(cls=OrderCommands, add_completion=False, no_args_is_help=True)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}

from .distributed.worker import worker_miss_align  # noqa: E402

cli.command(name="worker")(worker_miss_align)

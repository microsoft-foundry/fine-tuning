"""Replace CLI/training effects, not imports, in the real ``python -m`` recipe."""

import json
import os
import sys
from pathlib import Path


def worker_identity():
    return {
        "entry_file": str(Path(sys.modules["__main__"].__file__).resolve()),
        "training_modules": [
            name for name in sys.modules
            if name == "torch" or name == "datasets" or name.startswith("azure.")
            or name.endswith("train_azure") or "kimi_megatron" in name
        ],
    }


def install():
    import chz

    root = Path(os.environ["MATH_GRADING_SOURCE_ROOT"]).resolve()
    entry_path = root / "programmed_training/recipes/math_rl/train_azure.py"
    marker = Path(os.environ["MATH_GRADING_LAUNCH_MARKER"])

    def guard_entry(event, args):
        if event != "exec" or Path(args[0].co_filename).resolve() != entry_path:
            return
        with marker.open("a", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        if len(marker.read_text(encoding="utf-8").splitlines()) != 1:
            raise RuntimeError("Math grading worker re-entered the training entry module")

    sys.addaudithook(guard_entry)

    def offline_entrypoint(config_type):
        namespace = sys._getframe(1).f_globals
        assert namespace["__name__"] == "__main__"
        assert namespace["__spec__"].name == "programmed_training.recipes.math_rl.train_azure"
        assert Path(namespace["__file__"]).resolve() == entry_path
        # Module execution and its real Azure/Torch/datasets imports have
        # already happened. Replace only the external training effect.
        assert {"torch", "datasets", "azure.ai.finetuningsessions.models"} <= sys.modules.keys()
        math_env = namespace["math_env"]
        grader = sys.modules[math_env.grade_answer.__module__]
        assert Path(grader.__file__).resolve() == entry_path.with_name("math_grading.py")

        async def offline_cli_main(config):
            import socket

            from azure.ai.finetuningsessions.models import ModelInput

            def no_network(*args, **kwargs):
                raise AssertionError("The math module-launch regression must remain offline")

            # Install after asyncio has created its local wakeup sockets.
            socket.socket.connect = no_network
            socket.socket.connect_ex = no_network
            socket.getaddrinfo = no_network

            class Renderer:
                def parse_response(self, action):
                    return {"role": "assistant", "content": r"\boxed{0.5}"}, True

                def get_stop_sequences(self):
                    return []

            env = math_env.MathEnv("one half", "1/2", Renderer(), timeout=0.25)
            result = await env.step([])
            assert result.episode_done and result.reward == 1.0
            assert isinstance(result.next_observation, ModelInput)
            identity = grader.run_with_timeout_signal(worker_identity, timeout_seconds=0.25)
            assert identity["entry_file"] == str(entry_path.with_name("math_grading.py"))
            assert identity["training_modules"] == []
            assert len(marker.read_text(encoding="utf-8").splitlines()) == 1
            print("MATH_MODULE_LAUNCH_OK=" + json.dumps(identity), flush=True)

        namespace["cli_main"] = offline_cli_main
        return config_type(project_endpoint="https://offline.invalid")

    chz.entrypoint = offline_entrypoint
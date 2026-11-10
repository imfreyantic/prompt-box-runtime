"""llama.cpp / ComfyUI install helpers, exercised offline against local git repositories."""

import io
import json
import os
import shutil
import subprocess
import tarfile
import unittest
from pathlib import Path
from unittest import mock

from box import comfy_install as CI
from box import constants as C
from box import llama_build as LB
from box.procs import StepError
from box.sysinfo import effective_build_jobs, parse_gpu_query
from tests.helpers import PROMPT_BOX_DIR, tempdir

HAS_GIT = shutil.which("git") is not None


def git(*args, cwd=None) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@e")
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def make_source_repo(root: Path) -> tuple[Path, str, str]:
    src = root / "src"
    src.mkdir()
    git("init", "-q", cwd=src)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=src)
    (src / "model.cpp").write_text("int a = 1;\nint b = 2;\nint c = 3;\n")
    git("add", ".", cwd=src)
    git("commit", "-qm", "one", cwd=src)
    first = git("rev-parse", "HEAD", cwd=src)
    git("tag", "v0.44.0", cwd=src)
    (src / "model.cpp").write_text("int a = 1;\nint b = 2;\nint c = 4;\n")
    git("commit", "-qam", "two", cwd=src)
    second = git("rev-parse", "HEAD", cwd=src)
    return src, first, second


class ProgressAndJobsTests(unittest.TestCase):
    def test_parse_build_percent(self):
        self.assertEqual(LB.parse_build_percent("[12/48] Building CUDA object ggml/src/x.cu.o"), 25)
        self.assertEqual(LB.parse_build_percent("[48/48] Linking CXX executable bin/llama-server"), 100)
        self.assertEqual(LB.parse_build_percent("[ 42%] Building CXX object common/arg.cpp.o"), 42)
        self.assertEqual(LB.parse_build_percent("[100%] Built target llama-server"), 100)
        self.assertIsNone(LB.parse_build_percent("-- Configuring done"))

    def test_jobs_capped_by_cgroup_cpu_and_memory(self):
        with tempdir() as tmp:
            cg = Path(tmp)
            (cg / "cpu.max").write_text("800000 100000\n")  # 8 CPUs
            (cg / "memory.max").write_text(str(16 * 1024**3))  # 16 GiB -> 6 jobs at 2.5 GiB
            meminfo = cg / "meminfo"
            meminfo.write_text("MemTotal:       1056000000 kB\n")
            with mock.patch("os.sched_getaffinity", create=True, return_value=set(range(128))):
                self.assertEqual(effective_build_jobs(None, cg, meminfo), 6)
                (cg / "memory.max").write_text("max\n")
                self.assertEqual(effective_build_jobs(None, cg, meminfo), 8)
                (cg / "cpu.max").write_text("max 100000\n")
                self.assertEqual(effective_build_jobs(None, cg, meminfo), 128)
            self.assertEqual(effective_build_jobs(3, cg, meminfo), 3)

    def test_gpu_query_parse(self):
        csv = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 600.00, 600.00, 580.95.05\n"
        smi = "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0     |"
        self.assertEqual(parse_gpu_query(csv, smi), {
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "vramMiB": 97887,
            "powerLimitW": 600.0, "powerMaxW": 600.0, "driver": "580.95.05", "cuda": "13.0"})
        self.assertIsNone(parse_gpu_query(""))
        self.assertIsNone(parse_gpu_query("x, y"))
        na = parse_gpu_query("GPU, 1000, [N/A], [N/A], 1.0")
        self.assertIsNone(na["powerLimitW"])

    def test_build_marker_contents(self):
        self.assertEqual(LB.build_marker_expected(), {
            "commit": C.LLAMA_COMMIT, "patchSha256": C.PATCH_FILE.sha256,
            "cmakeArgs": list(C.LLAMA_CMAKE_ARGS), "target": "llama-server"})

    def test_build_env_prefers_cached_generator(self):
        with tempdir() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            (build / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
            env = LB.build_env(Path(tmp) / "venv" / "bin", build)
            self.assertEqual(env["CMAKE_GENERATOR"], "Unix Makefiles")
            self.assertTrue(env["PATH"].startswith(str(Path(tmp) / "venv" / "bin")))


@unittest.skipUnless(HAS_GIT, "git not installed")
class LlamaCheckoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempdir()
        self.root = Path(self._tmp.name)
        self.src, self.first, self.second = make_source_repo(self.root)
        self.log = self.root / "llama.log"
        self.repo = self.root / "llama.cpp"
        patcher = mock.patch.multiple(C, LLAMA_REPO_URL=f"file://{self.src}", LLAMA_COMMIT=self.first)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    async def test_checkout_exact_commit_and_idempotent(self):
        await LB.ensure_checkout(self.repo, self.log)
        self.assertEqual(git("rev-parse", "HEAD", cwd=self.repo), self.first)
        self.assertEqual((self.repo / "model.cpp").read_text().splitlines()[-1], "int c = 3;")
        (self.repo / "model.cpp").write_text("patched\n")  # local patch survives a re-run
        await LB.ensure_checkout(self.repo, self.log)
        self.assertEqual((self.repo / "model.cpp").read_text(), "patched\n")

    async def test_new_pin_moves_checkout_and_keeps_build_dir(self):
        await LB.ensure_checkout(self.repo, self.log)
        (self.repo / "build").mkdir()
        (self.repo / "build" / "keep.o").write_text("obj")
        (self.repo / "model.cpp").write_text("patched\n")
        with mock.patch.object(C, "LLAMA_COMMIT", self.second):
            await LB.ensure_checkout(self.repo, self.log)
        self.assertEqual(git("rev-parse", "HEAD", cwd=self.repo), self.second)
        self.assertTrue((self.repo / "build" / "keep.o").exists())

    async def test_refuses_foreign_directory(self):
        self.repo.mkdir()
        (self.repo / "notes.txt").write_text("mine")
        with self.assertRaises(StepError):
            await LB.ensure_checkout(self.repo, self.log)
        self.assertEqual((self.repo / "notes.txt").read_text(), "mine")

    async def test_patch_apply_is_idempotent(self):
        await LB.ensure_checkout(self.repo, self.log)
        patch = self.root / "fast.patch"
        patch.write_text(subprocess.run(["git", "diff", self.first, self.second], cwd=self.src,
                                        capture_output=True, text=True, check=True).stdout)
        self.assertEqual(await LB.ensure_patch_applied(self.repo, patch, self.log), "applied")
        self.assertEqual((self.repo / "model.cpp").read_text().splitlines()[-1], "int c = 4;")
        self.assertEqual(await LB.ensure_patch_applied(self.repo, patch, self.log), "already-applied")
        (self.repo / "model.cpp").write_text("something else entirely\n")
        with self.assertRaises(StepError):
            await LB.ensure_patch_applied(self.repo, patch, self.log)


@unittest.skipUnless(HAS_GIT, "git not installed")
class ComfyCheckoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempdir()
        self.root = Path(self._tmp.name)
        self.src, self.first, _ = make_source_repo(self.root)
        self.log = self.root / "comfy.log"
        patcher = mock.patch.multiple(C, COMFY_REPO_URL=f"file://{self.src}", COMFY_TAG="v0.44.0",
                                      COMFY_COMMIT=self.first)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    async def test_clone_tag_and_verify(self):
        dest = self.root / "ComfyUI"
        await CI.ensure_checkout(dest, self.log)
        self.assertEqual(git("rev-parse", "HEAD", cwd=dest), self.first)
        await CI.ensure_checkout(dest, self.log)  # idempotent

    async def test_repairs_interrupted_clone(self):
        dest = self.root / "ComfyUI"
        dest.mkdir()
        git("init", "-q", cwd=dest)
        (dest / "model.cpp").write_text("half written")
        await CI.ensure_checkout(dest, self.log)
        self.assertEqual(git("rev-parse", "HEAD", cwd=dest), self.first)
        self.assertEqual((dest / "model.cpp").read_text(), "int a = 1;\nint b = 2;\nint c = 3;\n")

    async def test_wrong_tag_commit_is_rejected(self):
        with mock.patch.object(C, "COMFY_COMMIT", "0" * 40):
            with self.assertRaises(StepError):
                await CI.ensure_checkout(self.root / "ComfyUI", self.log)


class InstallTests(unittest.TestCase):
    def test_custom_node_copied_without_bytecode(self):
        with tempdir() as tmp:
            comfy = Path(tmp) / "ComfyUI"
            dest = CI.install_custom_node(PROMPT_BOX_DIR, comfy)
            self.assertEqual(dest, comfy / "custom_nodes" / "aa_prompt_llm")
            self.assertTrue((dest / "__init__.py").is_file())
            self.assertTrue((dest / "nodes.py").is_file())
            self.assertFalse(list(dest.rglob("__pycache__")))

    def test_workflows_installed_and_user_edits_kept(self):
        with tempdir() as tmp:
            runtime = Path(tmp) / "runtime"
            (runtime / "workflows").mkdir(parents=True)
            (runtime / "workflows" / "A.json").write_text('{"v": 1}')
            (runtime / "workflows" / "B.json").write_text('{"v": 1}')
            comfy = Path(tmp) / "ComfyUI"
            record = Path(tmp) / "state" / "wf.json"
            dest = comfy / "user" / "default" / "workflows"

            self.assertEqual(CI.install_workflows(runtime, comfy, record), {"A.json": "installed", "B.json": "installed"})
            self.assertEqual(CI.install_workflows(runtime, comfy, record), {"A.json": "unchanged", "B.json": "unchanged"})
            (dest / "A.json").write_text('{"v": "user"}')  # user saved over it in ComfyUI
            (runtime / "workflows" / "A.json").write_text('{"v": 2}')
            (runtime / "workflows" / "B.json").write_text('{"v": 2}')
            actions = CI.install_workflows(runtime, comfy, record)
            self.assertEqual(actions, {"A.json": "kept-user-edit", "B.json": "updated"})
            self.assertEqual((dest / "A.json").read_text(), '{"v": "user"}')
            self.assertEqual((dest / "B.json").read_text(), '{"v": 2}')
            self.assertFalse(list(dest.glob(".*")))  # nothing hidden next to the workflows

    def test_shipped_workflows_land_in_user_folder(self):
        with tempdir() as tmp:
            comfy = Path(tmp) / "ComfyUI"
            actions = CI.install_workflows(PROMPT_BOX_DIR, comfy, Path(tmp) / "rec.json")
            self.assertEqual(set(actions), {"AA Prompt - Text to Prompt.json", "AA Prompt - Image to Prompt.json"})
            for name in actions:
                json.loads((comfy / "user" / "default" / "workflows" / name).read_text())


class PrebuiltTests(unittest.TestCase):
    def make_tar(self, path: Path, members: dict[str, bytes], prefix: str = "", links: dict[str, str] = None):
        with tarfile.open(path, "w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(prefix + name)
                info.size = len(data)
                info.mode = 0o755
                tar.addfile(info, io.BytesIO(data))
            for name, target in (links or {}).items():
                info = tarfile.TarInfo(prefix + name)
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)

    def test_nested_archive_with_libs(self):
        with tempdir() as tmp:
            archive = Path(tmp) / "llama.tgz"
            self.make_tar(archive, {"llama-server": b"#!bin", "libllama.so.0": b"lib"}, prefix="build/bin/",
                          links={"libllama.so": "libllama.so.0"})
            bin_dir = Path(tmp) / "llama.cpp" / "build" / "bin"
            server = LB.install_prebuilt_archive(archive, bin_dir)
            self.assertEqual(server, bin_dir / "llama-server")
            self.assertTrue(os.access(server, os.X_OK))
            self.assertEqual((bin_dir / "libllama.so").read_bytes(), b"lib")
            self.assertFalse(list(bin_dir.parent.glob(".prebuilt-staging-*")))

    def test_archive_without_server_rejected(self):
        with tempdir() as tmp:
            archive = Path(tmp) / "x.tgz"
            self.make_tar(archive, {"README": b"nothing"})
            with self.assertRaises(StepError):
                LB.install_prebuilt_archive(archive, Path(tmp) / "bin")

    def test_path_traversal_rejected(self):
        with tempdir() as tmp:
            archive = Path(tmp) / "evil.tgz"
            self.make_tar(archive, {"../../escape": b"x", "llama-server": b"x"})
            with self.assertRaises(Exception):
                LB.install_prebuilt_archive(archive, Path(tmp) / "a" / "bin")
            self.assertFalse((Path(tmp) / "escape").exists())


if __name__ == "__main__":
    unittest.main()

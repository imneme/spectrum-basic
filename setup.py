import os
from setuptools import setup
from setuptools.command.build_py import build_py

class GenerateAst(build_py):
    def run(self):
        # Run code generation before building
        self.run_code_generation()
        super().run()

    def run_code_generation(self):
        pkg_dir = "spectrum_basic"
        gen_ast_path = os.path.join(pkg_dir, "gen_ast.py")
        ast_py_path = os.path.join(pkg_dir, "ast.py")

        namespace = {'__file__': gen_ast_path}
        with open(gen_ast_path) as f:
            code = compile(f.read(), gen_ast_path, 'exec')
            exec(code, namespace)
        # call gen_ast_py function defined in gen_ast.py
        namespace['gen_ast_py'](ast_py_path)

setup(
    # Note: Most metadata is now specified in pyproject.toml
    # but we need at least packages and cmdclass here.
    packages=["spectrum_basic"],
    include_package_data=True,
    package_data={
        "spectrum_basic": ["*.tx"]
    },
    cmdclass={"build_py": GenerateAst},
)

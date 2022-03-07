from setuptools import setup

setup(name="echo-opt",
      version="0.1",
      description="Earth Computer Hyperparameter Optimization",
      author="John Schreck, David John Gagne, Charlie Becker, Gabrielle Gantos, Keely Lawrence",
      license="MIT",
      url="https://github.com/NCAR/echo-opt",
      packages=["echo", "echo/src"],
      entry_points = {
        'console_scripts': ['echo-opt=echo.optimize:main', 'echo-run=echo.run:main', 'echo-report=echo.report:main'],
      }
      )

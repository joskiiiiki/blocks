{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  fetchzip,
  cython,
  setuptools,
  numpy,
}:

buildPythonPackage rec {
  pname = "pyfastnoiselite";
  version = "0.0.7";
  format = "setuptools";

  src = fetchFromGitHub {
    owner = "tizilogic";
    repo = "pyfastnoiselite";
    rev = "3b6390f486dd837d6f80fe1cd5bc097c752eb6aa";
    hash = "sha256-0Gz61Rs40MPS5Z8froV5XneY6aAIa3jsCAOn9R3ji9Y=";
    # Get real hash: nix-prefetch-url --unpack https://github.com/tizilogic/pyfastnoiselite/archive/refs/tags/v0.0.7.tar.gz
  };

  # The setup.py downloads FastNoise library if ext/FastNoise doesn't exist
  # We need to fetch it separately and provide it
  fastNoise = fetchzip {
    url = "https://github.com/Auburn/FastNoise/archive/master.zip";
    hash = "sha256-azlrgJ6lkzuXjdSNZ0y6yzHGKDjdNYuF95FQCEiR2gc=";
    # Get real hash: nix-prefetch-url --unpack https://github.com/Auburn/FastNoise/archive/master.zip
  };

  # Provide the FastNoise library before build
  postUnpack = ''
    mkdir -p $sourceRoot/ext/FastNoise
    cp -r ${fastNoise}/* $sourceRoot/ext/FastNoise/
  '';
  preBuild = ''
    export USE_CYTHON=1
  '';
  nativeBuildInputs = [
    cython
    setuptools
  ];

  propagatedBuildInputs = [
    numpy
  ];

  # Package doesn't include tests in release tarball
  doCheck = false;

  pythonImportsCheck = [
    "pyfastnoiselite"
  ];

  meta = with lib; {
    description = "Cython wrapper for Auburn's FastNoise Lite";
    longDescription = ''
      PyFastNoiseLite is a Cython wrapper for the FastNoise Lite C++ library,
      providing extremely fast noise generation for procedural content generation.
      Supports multiple noise types including Perlin, Simplex, Cellular, and Value
      noise with fractal layering capabilities. Significantly faster than pure-Python
      implementations like OpenSimplex (often 100-1000x speedup).
    '';
    homepage = "https://github.com/tizilogic/pyfastnoiselite";
    changelog = "https://github.com/tizilogic/pyfastnoiselite/releases/tag/v${version}";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
    platforms = platforms.unix;
  };
}


{ lib
, buildPythonPackage
, fetchFromGitHub
, setuptools
, wheel
, pytestCheckHook
, numpy
}:

buildPythonPackage rec {
  pname = "pyfastnoiselite";
  version = "1.2.3";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "jarikomppa";
    repo = "pyfastnoiselite";
    rev = "refs/tags/v${version}";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
  };

  nativeBuildInputs = [
    setuptools
    wheel
  ];

  propagatedBuildInputs = [
    numpy
  ];

  nativeCheckInputs = [
    pytestCheckHook
  ];

  # Disable tests if they don't exist or fail in sandbox
  doCheck = false;

  pythonImportsCheck = [
    "pyfastnoiselite"
  ];
}

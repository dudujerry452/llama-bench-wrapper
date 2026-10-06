{
  description = "llama.cpp multi-request benchmarking workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llama = {
      url = "github:ggml-org/llama.cpp/46847e61582097979f539595d893d83d8e1d1af1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, llama, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.cudaSupport = true;
            config.allowUnfree = true;
          };
          llamaPackages = llama.packages.${system};
          trimPackage = package:
            (package.override { useWebUi = false; }).overrideAttrs (old: {
              webui = null;
              cmakeFlags = old.cmakeFlags ++ [
                "-DLLAMA_BUILD_TESTS:BOOL=FALSE"
                "-DLLAMA_BUILD_EXAMPLES:BOOL=FALSE"
                "-DLLAMA_BUILD_APP:BOOL=FALSE"
                "-DLLAMA_BUILD_TOOLS:BOOL=TRUE"
                "-DLLAMA_BUILD_COMMON:BOOL=TRUE"
                "-DLLAMA_BUILD_SERVER:BOOL=TRUE"
                "-DLLAMA_BUILD_UI:BOOL=FALSE"
              ];
            });
          cpuPackage = trimPackage llamaPackages.default;
          acceleratedPackage =
            if pkgs.stdenv.hostPlatform.isLinux then
              (trimPackage llamaPackages.cuda).overrideAttrs (old: {
                cmakeFlags = map (
                  flag:
                  if nixpkgs.lib.hasPrefix "-DCMAKE_CUDA_ARCHITECTURES" flag then
                    "-DCMAKE_CUDA_ARCHITECTURES:STRING=89"
                  else
                    flag
                ) old.cmakeFlags;
              })
            else
              cpuPackage;
        in
        {
          default = acceleratedPackage;
          cpu = cpuPackage;
          cuda = acceleratedPackage;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.cudaSupport = true;
            config.allowUnfree = true;
          };
          mkDevShell = package: pkgs.mkShell {
            name = "llama-things";
            packages = [
              package
              pkgs.python3
              pkgs.curl
              pkgs.git
              pkgs.jq
            ];
            shellHook = ''
              export LLAMA_THINGS_ROOT="$(pwd)"
              export LLAMA_CPP_ROOT="$LLAMA_THINGS_ROOT/llama.cpp"
              echo "llama-things devShell"
              echo "llama-server: $(command -v llama-server)"
              echo "llama-bench:  $(command -v llama-bench)"
            '';
          };
        in
        {
          default = mkDevShell self.packages.${system}.default;
          cuda = mkDevShell self.packages.${system}.cuda;
          cpu = mkDevShell self.packages.${system}.cpu;
        }
      );
    };
}

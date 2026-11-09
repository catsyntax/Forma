# Download Forma (ready to install)

**Current version: Forma 0.7.0** (adds Grasshopper import, 3D space syntax and the modular plan generator).

Pick your computer and click the link. The download starts straight away. These are complete apps: you
don't need Python or anything else.

| Your computer | Download | Size |
|---|---|---|
| **Mac with Apple chip** (M1, M2, M3, M4…) | [Forma-macOS-AppleSilicon.dmg](https://github.com/catsyntax/Forma/releases/latest/download/Forma-macOS-AppleSilicon.dmg) | about 320 MB |
| **Mac with Intel chip** | [Forma-macOS-Intel.dmg](https://github.com/catsyntax/Forma/releases/latest/download/Forma-macOS-Intel.dmg) | about 340 MB |
| **Windows 10 / 11** | [Forma-Windows.zip](https://github.com/catsyntax/Forma/releases/latest/download/Forma-Windows.zip) | about 260 MB |
| **Linux** (Ubuntu 22.04 or newer, Fedora, …) | [Forma-Linux.tar.gz](https://github.com/catsyntax/Forma/releases/latest/download/Forma-Linux.tar.gz) | about 380 MB |

Not sure which Mac you have? Apple menu → **About This Mac**: "Chip Apple M…" means Apple chip,
"Processor Intel…" means Intel.

The app files are too large to store inside the repository itself (GitHub allows 100 MB per file), so they
live on the [Releases page](https://github.com/catsyntax/Forma/releases/latest). The links above always
point to the newest version.

## Install

### Mac
1. Open the downloaded `.dmg` and drag **Forma** onto **Applications**.
2. The first time, **right-click Forma → Open → Open** (the app isn't signed with a paid Apple certificate,
   so macOS asks once). If macOS says the app "is damaged", open Terminal and run
   `xattr -dr com.apple.quarantine /Applications/Forma.app`, then open it again.

### Windows
1. Right-click the downloaded `.zip` → **Extract All…**, and put the `Forma` folder somewhere permanent
   (for example `C:\Program Files\Forma` or your Documents).
2. Double-click **Forma.exe**. If Windows shows "Windows protected your PC": **More info → Run anyway**.
3. Optional: right-click Forma.exe → **Pin to Start**.

### Linux
1. Unpack it: `tar xzf Forma-Linux.tar.gz`
2. Start it: `./Forma/Forma`  (Ubuntu/Debian may first need `sudo apt install libxcb-cursor0 libgl1`)
3. Optional, adds Forma to your applications menu: `./Forma/install.sh`

## Updating
Download the new file and install it the same way. On Mac, drag the new Forma onto Applications and choose
**Replace**; on Windows and Linux, replace the old `Forma` folder. Your settings and shortcuts are kept
(they live in a `.forma` folder in your home folder, not inside the app).

## First steps
- **File → Open** an example from the `examples` folder of this repository, or start drawing: type a
  command such as `Box` or `Line` and press Enter. **F1** lists every command.
- **Nodes** tab: load a graph from **Analysis examples**, or click **Import Grasshopper…** to bring in a
  `.gh` / `.ghx` definition.

Prefer to run from the source code instead? See [INSTALL.md](../INSTALL.md).

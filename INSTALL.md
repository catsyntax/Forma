# Installing Forma

**Easiest:** download the ready-made app for your computer from **[Download App](Download%20App/README.md)**.
No Python, no setup. The rest of this page is for running Forma from its source code instead.

Forma runs on **macOS**, **Windows** and **Linux**. You don't need to install Python or anything else first:
the start file sets everything up the first time (about 5–10 minutes and a 1 GB download), then Forma opens in
a few seconds every time after that. Everything goes into a `.venv` folder inside the Forma folder, so deleting
the folder removes Forma completely.

## 1. Download

1. On the GitHub page, click the green **Code** button, then **Download ZIP**.
2. Unzip it. Put the `forma` folder somewhere you'll keep it, for example in Documents.

(Or, if you use git: `git clone https://github.com/catsyntax/Forma.git`)

## 2. Start Forma

### macOS

1. Open the `forma` folder and double-click **`start_mac.command`**.
2. If macOS says it can't be opened because it's from an unidentified developer:
   right-click the file, choose **Open**, then **Open** again. (You only need to do this once.)
3. A Terminal window opens and shows the setup progress. Leave it open; Forma appears when setup is done.
4. Optional: to get a proper app for your Dock, double-click `packaging/make_mac_app.command` after the
   first start. It creates **Forma.app** (and moves the set-up libraries into it).

### Windows

1. Open the `forma` folder and double-click **`start_windows.bat`**.
2. If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**.
3. A black window shows the setup progress. Leave it open; Forma appears when setup is done.
4. Next time you can start Forma from the **Start menu** (a Forma shortcut is added during setup),
   or by double-clicking `start_windows.bat` again.

### Linux

1. Install one system library that Qt needs (once):
   - Ubuntu / Debian / Mint: `sudo apt install libxcb-cursor0 libgl1`
   - Fedora: `sudo dnf install xcb-util-cursor mesa-libGL`
   - Arch: `sudo pacman -S xcb-util-cursor`
2. Open a terminal in the `forma` folder and run: `./start_linux.sh`
   (if it says "permission denied", run `chmod +x start_linux.sh` first).
3. When setup is done Forma opens, and a **Forma** entry is added to your applications menu.

## 3. First steps

1. **File → Open** → `examples/sample_project.forma` to see a project with drawings, NURBS solids,
   components and an analysis graph.
2. Switch workspaces with the tabs at the top: **Draft**, **Model**, **Components**, **Nodes**
   (or Ctrl/Cmd + 1, 2, 3, 4).
3. Type a command anywhere (for example `Box`, `Line`, `Extrude`, `Trim`) and press Enter.
   **F1** lists every command and shortcut.
4. In **Nodes**, pick an item from **Analysis examples** in the toolbar to see daylight, shadows,
   solar gains, energy and space syntax in action.

## Updating

Download the new version and replace the files in your `forma` folder (keep the `.venv` folder to skip the
long setup). If the list of libraries changed, the start file installs the new ones automatically.

## Problems?

| Problem | Fix |
|---|---|
| Setup stops with a download error | Check your internet connection and run the start file again. It continues where it stopped. |
| macOS: "cannot be opened" | Right-click the file → **Open** → **Open**. |
| macOS: "permission denied" | In Terminal, in the folder: `chmod +x start_mac.command` |
| Windows: the window flashes and closes | Open the folder in File Explorer, type `cmd` in the address bar and press Enter, then type `start_windows.bat` and press Enter to read the messages. |
| Linux: "could not load the Qt platform plugin xcb" | Install the library from Linux step 1. |
| Forma closed unexpectedly | Run the start file again from a terminal and keep the messages; they show what went wrong. |
| Start fresh | Delete the `.venv` folder inside the Forma folder and run the start file again. |

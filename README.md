# GTA V plugin for gamehook

This plugin depends to [gamehook](https://github.com/philkr/gamehook), and is compatible with [pyhookv](https://github.com/philkr/pyhookv).
It extracts many ground truth modalities from GTA V, which can then be saved using the server or capture plugins.

To use this plugin copy (or SYMLINK) the SDK headers and gamehook.lib files from the main gamehook directory. Then download and copy the scripthook files.
The solution should produce a gta5.hk file, which is automatically loaded by gamehook when copies (or SYMLINKED) into the GTA V directory.

The code is currently a bit messy.
It uses scripthook to track objects (compared to my own tracker for used in the main paper). The scripthook tracker is a bit more relyable.

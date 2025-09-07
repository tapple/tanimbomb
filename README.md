# tanimbomb
Library to parse, manipulate, create, and export Secondlife animation files.

Includes a command line tool to edit Secondlife animation files in bulk

To install:
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `uv tool install git+https://github.com/tapple/tanimbomb[quaternion]`

View animation info:
```
$ animDump *.anim
DogAngry.anim             : P6 39R  0L (0.00m) 0C 0.8-0.8Es  0.000s   looped (0.00in +  0.00 +  0.00out)
Dog_Dance_1.anim          : P6 97R  1L (0.06m) 0C 0.8-0.1Es  2.148s   looped (0.00in +  2.15 +  0.00out) at 27fps (  57 frames)
Jaw5.anim                 : P6  1R  0L (0.00m) 0C 0.8-0.8Es  0.000s   looped (0.00in +  0.00 +  0.00out)
PantingLight.anim         : P5 12R  0L (0.00m) 0C 0.8-0.8Es  2.360s   looped (0.00in +  2.36 +  0.00out) at 25fps (  58 frames)
PantingSide 1.anim        : P3 12R  0L (0.00m) 0C 0.8-0.8Es  0.040s unlooped (0.00in +  0.00 +  0.04out) at 25fps (   0 frames)
PantingSide 2.anim        : P5 12R  0L (0.00m) 0C 0.8-0.8Es  2.360s   looped (0.00in +  2.36 +  0.00out) at 25fps (  58 frames)
```

View animation info (markdown):
```
$ animDump --md *.anim
```
|Filename|Pri|Rots|Locs|Range|Cons|E In|E Out|Dur|Loop|L In|L Dur|L Out|FPS|Frames|
|--------|--:|---:|---:|----:|---:|---:|----:|--:|---:|---:|----:|----:|--:|-----:|
|DogAngry.anim             |6|39|  0|0.00|  0|0.8|0.8| 0.0000|  looped| 0.00| 0.0000| 0.00|
|Dog_Dance_1.anim          |6|97|  1|0.06|  0|0.8|0.1| 2.1481|  looped| 0.00| 2.1481| 0.00|27|  57|
|Jaw5.anim                 |6| 1|  0|0.00|  0|0.8|0.8| 0.0000|  looped| 0.00| 0.0000| 0.00|
|Jaw6.anim                 |6| 1|  0|0.00|  0|0.8|0.8| 0.0000|  looped| 0.00| 0.0000| 0.00|
|PantingLight.anim         |5|12|  0|0.00|  0|0.8|0.8| 2.3600|  looped| 0.00| 2.3600| 0.00|25|  58|
|PantingSide 1.anim        |3|12|  0|0.00|  0|0.8|0.8| 0.0400|unlooped| 0.00| 0.0000| 0.04|25|   0|
|PantingSide 2.anim        |5|12|  0|0.00|  0|0.8|0.8| 2.3600|  looped| 0.00| 2.3600| 0.00|25|  58|


Bulk-edit animation priority:
```
$ animDump wolf_stand* --pri 3 -o ./%n_av
wolf_stand1_av.anim: P3 97R  1L (0.06m) 0C 0.8-0.8Es 27.722s   looped (0.00in + 27.72 +  0.00out) at 18fps ( 498 frames)
wolf_stand2_av.anim: P3 97R  1L (0.01m) 0C 0.8-0.8Es 27.722s   looped (0.00in + 27.72 +  0.00out) at 18fps ( 498 frames)
wolf_stand3_av.anim: P3 43R  1L (0.23m) 0C 0.8-0.8Es 24.950s   looped (0.00in + 24.95 +  0.00out) at 20fps ( 499 frames)
```

Bulk-mirror animations:
```
$ animDump *.anim --mirror -o mirrored/%n
mirrored/Stand-Haunches-Left.anim       : P4 25R  1L (0.00m) 0C 0.3-0.3Es  2.633s   looped (0.00in + 26.63 + -24.00out) at  1fps (   2 frames)
mirrored/Stand-Leg Yield-Left.anim      : P4 25R  1L (0.00m) 0C 0.3-0.3Es  2.633s   looped (0.00in + 26.63 + -24.00out) at  1fps (   2 frames)
mirrored/Stand-Shoulder-Right.anim      : P4 25R  1L (0.00m) 0C 0.3-0.3Es  2.667s   looped (0.03in +  2.63 +  0.00out) at 30fps (  80 frames)
```

All animation manipulations:
```
$ animDump --help
usage: animDump [-h] [--verbose] [--unordered] [--markdown] [--outputfile-pattern OUTPUTFILE_PATTERN] [--time-scale ACTIONS] [--frame-rate ACTIONS] [--offset ACTIONS [ACTIONS ...]] [--loc ACTIONS [ACTIONS ...]] [--rotate ACTIONS [ACTIONS ...]]
                [--scale [ACTIONS ...]] [--extend ACTIONS] [--delay ACTIONS] [--append ACTIONS] [--prepend ACTIONS] [--trim--rtrim ACTIONS] [--mirror] [--sort] [--joint-pri ACTIONS ACTIONS] [--pri ACTIONS] [--ease-in ACTIONS] [--ease-out ACTIONS]
                [--loop ACTIONS] [--loop-start ACTIONS] [--loop-end ACTIONS] [--freeze [ACTIONS ...]] [--drop-loc [ACTIONS ...]] [--drop-rot ACTIONS [ACTIONS ...]] [--drop-pri [ACTIONS ...]] [--drop-joint ACTIONS [ACTIONS ...]] [--drop-empty-joints]
                [--add-constraint ACTIONS ACTIONS ACTIONS] [--drop-constraints] [--c-plane] [--c-ease ACTIONS ACTIONS ACTIONS ACTIONS] [--c-source-offset ACTIONS ACTIONS ACTIONS] [--c-target-offset ACTIONS ACTIONS ACTIONS]
                [--c-target-dir ACTIONS ACTIONS ACTIONS]
                files [files ...]

Manipulate Secondlife .anim files

positional arguments:
  files                 anim files to dump or process

options:
  -h, --help            show this help message and exit
  --verbose, -v
  --unordered, -U       when printing joints with -v, show in file order rather than abc order
  --markdown, --md      output in markdown table
  --outputfile-pattern OUTPUTFILE_PATTERN, -o OUTPUTFILE_PATTERN
                        Output anim file path/name, with template substitution:
                            %n: input file name
                            %p: input file directory
                            %%: literal '%'
                        File extension will be appended automatically
  --time-scale ACTIONS, --tscale ACTIONS, --speed ACTIONS, -s ACTIONS
                        Adjust duration by the given factor eg:
                            2.0 for half-speed/double duration, or
                            0.5 for double speed/half duration
  --frame-rate ACTIONS, --fps ACTIONS
  --offset ACTIONS [ACTIONS ...], --adjust ACTIONS [ACTIONS ...]
                        Adjust joint location on all the given joint patterns (mPelvis by default). Examples:
                            "--offset 0.5z": move mPelvis up 0.5
                            "--offset mFaceNose* -0.2y 0.1x": move nose bones right 0.2m, forward 0.1m
  --loc ACTIONS [ACTIONS ...]
                        Move joint location on all keyframes so the starting location is at the given coordinates. Specify x y z to select which coordinates to set; missing axes will remain unchanged. Joint is optional, and defaults to mPelvis. Example:
                            "--loc mPelvis* 0x 0.25z": Move the animation so that it starts at 0 on x and 0.25 on z. Leave y alone
  --rotate ACTIONS [ACTIONS ...], --rot ACTIONS [ACTIONS ...]
                        Rotate each given joint (or mPelvis by default) on the given axes the given joint patterns. Examples:
                            "--rot 45z": rotate mPelvis 45° left around z
                            "--rot mFaceNose* -20y mFaceJaw 30y": rotate nose bones 20° up; rotate jaw 30° down
  --scale [ACTIONS ...]
                        Scale location keys; eg 2.0 for double-size avatar, 0.5 for half-size avatar. Can specify joint patterns or x y z to control individual joints
  --extend ACTIONS      Hold the final pose for the given number of seconds in the outro of the animation
  --delay ACTIONS       Add the given number of seconds to the intro of the animation
  --append ACTIONS      Add keyframes another anim file to the end of the ease out
  --prepend ACTIONS     Add keyframes another anim file to the beginning of the ease in
  --trim--rtrim ACTIONS, --trunc ACTIONS, --rtrunc ACTIONS
                        Remove the end of the animation starting at the given time
  --mirror, --flip
  --sort                Sort the joints by name in the output files
  --joint-pri ACTIONS ACTIONS
  --pri ACTIONS
  --ease-in ACTIONS
  --ease-out ACTIONS
  --loop ACTIONS
  --loop-start ACTIONS
  --loop-end ACTIONS
  --freeze [ACTIONS ...], --static [ACTIONS ...]
                        Freeze the given joint patterns by removing all keyframes except the first (if no joints given, freeze all joints)
  --drop-loc [ACTIONS ...]
                        Drop location keyframes from the given joint patterns (if none specified, drop all except mPelvis)
  --drop-rot ACTIONS [ACTIONS ...]
                        Drop rotation keyframes from the given joint patterns
  --drop-pri [ACTIONS ...]
                        Drop per-joint priority from the given joint patterns (if none specified, drop from all joints)
  --drop-joint ACTIONS [ACTIONS ...]
                        Drop the given joint patterns
  --drop-empty-joints   Drop all joints that have 0 rotation keyframes and 0 location keyframes
  --add-constraint ACTIONS ACTIONS ACTIONS
  --drop-constraints
  --c-plane
  --c-ease ACTIONS ACTIONS ACTIONS ACTIONS
  --c-source-offset ACTIONS ACTIONS ACTIONS
  --c-target-offset ACTIONS ACTIONS ACTIONS
  --c-target-dir ACTIONS ACTIONS ACTIONS

```

Similar projects:
- [AnimMaker](https://github.com/LGGGreg/par/releases/tag/v1) in C#
- [AnimHacker](https://aiaicapta.in/anim-hacker/) has a UI

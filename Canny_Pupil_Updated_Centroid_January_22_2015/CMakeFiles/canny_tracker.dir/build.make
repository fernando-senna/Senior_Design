# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid

# Include any dependencies generated for this target.
include CMakeFiles/canny_tracker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/canny_tracker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/canny_tracker.dir/flags.make

CMakeFiles/canny_tracker.dir/canny_main.cpp.o: CMakeFiles/canny_tracker.dir/flags.make
CMakeFiles/canny_tracker.dir/canny_main.cpp.o: canny_main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/canny_tracker.dir/canny_main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/canny_tracker.dir/canny_main.cpp.o -c /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/canny_main.cpp

CMakeFiles/canny_tracker.dir/canny_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/canny_tracker.dir/canny_main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/canny_main.cpp > CMakeFiles/canny_tracker.dir/canny_main.cpp.i

CMakeFiles/canny_tracker.dir/canny_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/canny_tracker.dir/canny_main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/canny_main.cpp -o CMakeFiles/canny_tracker.dir/canny_main.cpp.s

CMakeFiles/canny_tracker.dir/canny_main.cpp.o.requires:
.PHONY : CMakeFiles/canny_tracker.dir/canny_main.cpp.o.requires

CMakeFiles/canny_tracker.dir/canny_main.cpp.o.provides: CMakeFiles/canny_tracker.dir/canny_main.cpp.o.requires
	$(MAKE) -f CMakeFiles/canny_tracker.dir/build.make CMakeFiles/canny_tracker.dir/canny_main.cpp.o.provides.build
.PHONY : CMakeFiles/canny_tracker.dir/canny_main.cpp.o.provides

CMakeFiles/canny_tracker.dir/canny_main.cpp.o.provides.build: CMakeFiles/canny_tracker.dir/canny_main.cpp.o

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o: CMakeFiles/canny_tracker.dir/flags.make
CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o: PupilTracker.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o -c /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/PupilTracker.cpp

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/canny_tracker.dir/PupilTracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/PupilTracker.cpp > CMakeFiles/canny_tracker.dir/PupilTracker.cpp.i

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/canny_tracker.dir/PupilTracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/PupilTracker.cpp -o CMakeFiles/canny_tracker.dir/PupilTracker.cpp.s

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.requires:
.PHONY : CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.requires

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.provides: CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/canny_tracker.dir/build.make CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.provides.build
.PHONY : CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.provides

CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.provides.build: CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o

# Object files for target canny_tracker
canny_tracker_OBJECTS = \
"CMakeFiles/canny_tracker.dir/canny_main.cpp.o" \
"CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o"

# External object files for target canny_tracker
canny_tracker_EXTERNAL_OBJECTS =

canny_tracker: CMakeFiles/canny_tracker.dir/canny_main.cpp.o
canny_tracker: CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o
canny_tracker: CMakeFiles/canny_tracker.dir/build.make
canny_tracker: /usr/local/lib/libopencv_videostab.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_video.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_ts.a
canny_tracker: /usr/local/lib/libopencv_superres.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_stitching.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_photo.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_ocl.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_objdetect.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_nonfree.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_ml.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_legacy.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_imgproc.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_highgui.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_gpu.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_flann.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_features2d.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_core.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_contrib.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_calib3d.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_nonfree.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_ocl.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_gpu.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_photo.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_objdetect.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_legacy.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_video.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_ml.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_calib3d.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_features2d.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_highgui.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_imgproc.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_flann.so.2.4.12
canny_tracker: /usr/local/lib/libopencv_core.so.2.4.12
canny_tracker: /usr/local/lib/libtbb.so
canny_tracker: CMakeFiles/canny_tracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable canny_tracker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/canny_tracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/canny_tracker.dir/build: canny_tracker
.PHONY : CMakeFiles/canny_tracker.dir/build

CMakeFiles/canny_tracker.dir/requires: CMakeFiles/canny_tracker.dir/canny_main.cpp.o.requires
CMakeFiles/canny_tracker.dir/requires: CMakeFiles/canny_tracker.dir/PupilTracker.cpp.o.requires
.PHONY : CMakeFiles/canny_tracker.dir/requires

CMakeFiles/canny_tracker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/canny_tracker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/canny_tracker.dir/clean

CMakeFiles/canny_tracker.dir/depend:
	cd /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid /home/cseflnsn22/Desktop/Canny_Pupil_Updated_Centroid/CMakeFiles/canny_tracker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/canny_tracker.dir/depend


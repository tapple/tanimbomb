# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8-80 compliant>

bl_info = {
    "name": "SecondLife Animation (ANIM) format",
    "author": "Tapple Gao",
    "version": (2021, 4, 10),
    "blender": (2, 74, 0),
    "location": "File > Import-Export",
    "description": "Import SecondLife anim files to Avastar rigs",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/"
                "Scripts/Import-Export/BVH_Importer_Exporter",
    "support": 'COMMUNITY',
    "category": "Import-Export"}

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import (
        ImportHelper,
        ExportHelper,
        orientation_helper_factory,
        axis_conversion,
        )


ImportBVHOrientationHelper = orientation_helper_factory("ImportBVHOrientationHelper", axis_forward='-Z', axis_up='Y')


class ImportANIM(bpy.types.Operator, ImportHelper):
    """Load a BVH motion capture file"""
    bl_idname = "import_anim.anim"
    bl_label = "Import SecondLife Anim"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".anim"
    filter_glob = StringProperty(default="*.anim", options={'HIDDEN'})

    def execute(self, context):
        print(self.as_keywords())
        # from . import import_anim
        # return import_anim.load(context, report=self.report, **keywords)


def menu_func_import(self, context):
    self.layout.operator(ImportANIM.bl_idname, text="SecondLife Animation (.anim)")


def register():
    bpy.utils.register_module(__name__)

    bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_module(__name__)

    bpy.types.INFO_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()

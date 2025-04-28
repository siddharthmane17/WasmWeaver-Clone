from typing import List, Type
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
import numpy as np
from core.tile import AbstractTile

MAX_TILE_ID = 512
MAX_ACTIONS = 300
MAX_ARG_VALUE = 128

class TilesEmbedder:

    def __init__(self):
        ...

    def get_space(self):
        return Dict({
            "id":Discrete(MAX_TILE_ID),
            "args":Discrete(MAX_ARG_VALUE)
        })

    def __call__(self, tile: AbstractTile):
        return {"id":self.get_id(tile), "args":self.get_args(tile)}


    def get_id(self, tile: AbstractTile | Type[AbstractTile]):
        match tile.name:
            #Phantom tiles
            case "Finish":
                return 5

            #Basic tiles
            case "NoOp":
                return 10
            case "Drop":
                return 11
            case "Select":
                return 12

            #Memory tiles
            case "Create and call function":
                return 30
            case "Call function":
                return 31
            case "Indirect call function":
                return 32
            case "Push function reference to stack":
                return 33

            #Block tiles
            case "Create block":
                return 40

            #Conditional tiles
            case "Create conditional":
                return 50
            case "Create unbounded loop":
                return 60
            case "Create bounded loop":
                return 61

            #Debug tiles
            case "Canary":
                return 70

            #F32 tiles
            case "F32Const":
                return 80
            case "F32Add":
                return 81
            case "F32Sub":
                return 82
            case "F32Mul":
                return 83
            case "F32Div":
                return 84
            case "F32Sqrt":
                return 85
            case "F32Min":
                return 86
            case "F32Max":
                return 87
            case "F32Ceil":
                return 88
            case "F32Floor":
                return 89
            case "F32Trunc":
                return 90
            case "F32Nearest":
                return 91
            case "F32Abs":
                return 92
            case "F32Neg":
                return 93
            case "F32CopySign":
                return 94
            case "F32Eq":
                return 95
            case "F32Ne":
                return 96
            case "F32Lt":
                return 97
            case "F32Le":
                return 98
            case "F32Gt":
                return 99
            case "F32Ge":
                return 100
            case "F32DemoteF64":
                return 101
            case "F32ConvertI32S":
                return 102
            case "F32ConvertI32U":
                return 103
            case "F32ConvertI64S":
                return 104
            case "F32ConvertI64U":
                return 105
            case "F32ReinterpretI32":
                return 106
            case "F32Store":
                return 107
            case "F32Load":
                return 108

            #F64 tiles
            case "F64Const":
                return 120
            case "F64Add":
                return 121
            case "F64Sub":
                return 122
            case "F64Mul":
                return 123
            case "F64Div":
                return 124
            case "F64Sqrt":
                return 125
            case "F64Min":
                return 126
            case "F64Max":
                return 127
            case "F64Ceil":
                return 128
            case "F64Floor":
                return 129
            case "F64Trunc":
                return 130
            case "F64Nearest":
                return 131
            case "F64Abs":
                return 132
            case "F64Neg":
                return 133
            case "F64CopySign":
                return 134
            case "F64Eq":
                return 135
            case "F64Ne":
                return 136
            case "F64Lt":
                return 137
            case "F64Le":
                return 138
            case "F64Gt":
                return 139
            case "F64Ge":
                return 140
            case "F64PromoteF32":
                return 141
            case "F64ConvertI32S":
                return 142
            case "F64ConvertI32U":
                return 143
            case "F64ConvertI64S":
                return 144
            case "F64ConvertI64U":
                return 145
            case "F64ReinterpretI64":
                return 146
            case "F64Store":
                return 147
            case "F64Load":
                return 148

            #I32 tiles
            case "I32Const":
                return 160
            case "I32Add":
                return 161
            case "I32Sub":
                return 162
            case "I32Mul":
                return 163
            case "I32DivS":
                return 164
            case "I32DivU":
                return 165
            case "I32RemS":
                return 166
            case "I32RemU":
                return 167
            case "I32And":
                return 168
            case "I32Or":
                return 169
            case "I32Xor":
                return 170
            case "I32Shl":
                return 171
            case "I32ShrS":
                return 172
            case "I32ShrU":
                return 173
            case "I32Rotl":
                return 174
            case "I32Rotr":
                return 175
            case "I32Clz":
                return 176
            case "I32Ctz":
                return 177
            case "I32Popcnt":
                return 178
            case "I32Eqz":
                return 179
            case "I32Eq":
                return 180
            case "I32Ne":
                return 181
            case "I32LtS":
                return 182
            case "I32LtU":
                return 183
            case "I32LeS":
                return 184
            case "I32LeU":
                return 185
            case "I32GtS":
                return 186
            case "I32GtU":
                return 187
            case "I32GeS":
                return 188
            case "I32GeU":
                return 189
            case "I32WrapI64":
                return 190
            case "I32TruncF32S":
                return 191
            case "I32TruncF64S":
                return 192
            case "I32TruncF32U":
                return 193
            case "I32TruncF64U":
                return 194
            case "I32ReinterpretF32":
                return 195
            case "I32Extend8S":
                return 196
            case "I32Extend16S":
                return 197
            case "I32Store":
                return 198
            case "I32Store8":
                return 199
            case "I32Store16":
                return 200
            case "I32Load":
                return 201
            case "I32Load8U":
                return 202
            case "I32Load8S":
                return 203
            case "I32Load16U":
                return 204
            case "I32Load16S":
                return 205

            #I64 tiles
            case "I64Const":
                return 220
            case "I64Add":
                return 221
            case "I64Sub":
                return 222
            case "I64Mul":
                return 223
            case "I64DivS":
                return 224
            case "I64DivU":
                return 225
            case "I64RemS":
                return 226
            case "I64RemU":
                return 227
            case "I64And":
                return 228
            case "I64Or":
                return 229
            case "I64Xor":
                return 230
            case "I64Shl":
                return 231
            case "I64ShrS":
                return 232
            case "I64ShrU":
                return 233
            case "I64Rotl":
                return 234
            case "I64Rotr":
                return 235
            case "I64Clz":
                return 236
            case "I64Ctz":
                return 237
            case "I64Popcnt":
                return 238
            case "I64Eqz":
                return 239
            case "I64Eq":
                return 240
            case "I64Ne":
                return 241
            case "I64LtS":
                return 242
            case "I64LtU":
                return 243
            case "I64LeS":
                return 244
            case "I64LeU":
                return 245
            case "I64GtS":
                return 246
            case "I64GtU":
                return 247
            case "I64GeS":
                return 248
            case "I64GeU":
                return 249
            case "I64ExtendI32S":
                return 250
            case "I64ExtendI32U":
                return 251
            case "I64TruncF32S":
                return 252
            case "I64TruncF64S":
                return 253
            case "I64TruncF32U":
                return 254
            case "I64TruncF64U":
                return 255
            case "I64ReinterpretF64":
                return 256
            case "I64Extend8S":
                return 257
            case "I64Extend16S":
                return 258
            case "I64Extend32S":
                return 259
            case "I64Store":
                return 260
            case "I64Store8":
                return 261
            case "I64Store16":
                return 262
            case "I64Store32":
                return 263
            case "I64Load":
                return 264
            case "I64Load8U":
                return 265
            case "I64Load8S":
                return 266
            case "I64Load16U":
                return 267
            case "I64Load16S":
                return 268
            case "I64Load32U":
                return 269
            case "I64Load32S":
                return 270

            # Locals
            case "Get local":
                return 280
            case "Set local":
                return 281
            case "Tee local":
                return 282

            # Globals
            case "Get global":
                return 290
            case "Set global":
                return 291

            # Memory
            case "Memory size":
                return 300

            # Tables
            case "Get table":
                return 310
            case "Set table":
                return 311

            case _:
                raise ValueError(f"Unknown tile name: {tile.name}")
    def get_args(self, tile: AbstractTile | Type[AbstractTile]):
        match tile.name:
            #Functions
            case "Create and call function":
                return tile.index
            case "Call function":
                return tile.index
            case "Indirect call function":
                return tile.index
            case "Push function reference to stack":
                return tile.index

            #Globals
            case "Get global":
                return tile.index
            case "Set global":
                return tile.index

            #Locals
            case "Get local":
                return tile.index
            case "Set local":
                return tile.index
            case "Tee local":
                return tile.index

            #Tables
            case "Get table":
                return tile.index
            case "Set table":
                return tile.index

            case _:
                return 0

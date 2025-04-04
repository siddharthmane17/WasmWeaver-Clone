# WasmWeaver
Is used for generating the dataset for WATMeter models. The generator generates **correct** code which will run to completion without exceptions.
Further, soft lower response time/fuel/byte code limits and soft upper limits are used during generation to ensure that the generated code is within the bounds of the model (most of the time).

#### Supported Instructions

At the current time we do support most instructions and concepts. The following instructions are already supported:
##### Control Flow Instructions
- [x] block
- [x] loop: Only bounded for loops are supported
- [ ] br, br_if, br_table: Are contained in the loop instruction, but not used directly
- [x] if, else
- [ ] return: Not supported
- [ ] unreachable: Not supported

##### Basic Instructions
- [x] nop
- [x] drop
- [x] const
- [x] local.get, local.set, local.tee
- [x] global.get, global.set
- [x] select
- [x] call
- [ ] call_indirect: Not yet supported

##### Integer Arithmetic Instructions

- [x] i32/i64.add
- [x] i32/i64.sub
- [x] i32/i64.mul
- [x] i32/i64.div_s
- [x] i32/i64.div_u
- [x] i32/i64.rem_s
- [x] i32/i64.rem_u
- [x] i32/i64.and
- [x] i32/i64.or
- [x] i32/i64.xor
- [x] i32/i64.shl
- [x] i32/i64.shr_s
- [x] i32/i64.shr_u
- [x] i32/i64.rotl
- [x] i32/i64.rotr
- [x] i32/i64.clz
- [x] i32/i64.ctz
- [x] i32/i64.popcnt
- [x] i32/i64.eqz
##### Floating Point Arithmetic Instructions
- [x] f32/f64.add
- [x] f32/f64.sub
- [x] f32/f64.mul
- [x] f32/f64.div
- [x] f32/f64.sqrt
- [x] f32/f64.min
- [x] f32/f64.max
- [x] f32/f64.ceil
- [x] f32/f64.floor
- [x] f32/f64.trunc
- [x] f32/f64.nearest
- [x] f32/f64.abs
- [x] f32/f64.neg
- [x] f32/f64.copysign
##### Integer Comparison Instructions
- [x] i32/i64.eq
- [x] i32/i64.ne
- [x] i32/i64.lt_s
- [x] i32/i64.lt_u
- [x] i32/i64.le_s
- [x] i32/i64.le_u
- [x] i32/i64.gt_s
- [x] i32/i64.gt_u
- [x] i32/i64.ge_s
- [x] i32/i64.ge_u

##### Floating Point Comparison Instructions
- [x] f32/f64.eq
- [x] f32/f64.ne
- [x] f32/f64.lt
- [x] f32/f64.le
- [x] f32/f64.gt
- [x] f32/f64.ge

##### Conversion Instructions
- [x] i32.wrap_i64
- [x] i64.extend_i32_s
- [x] i64.extend_i32_u
- [x] i32.trunc_f32_s
- [x] i32.trunc_f32_u
- [x] i32.trunc_f64_s
- [x] i32.trunc_f64_u
- [x] f32.demote_f64
- [x] f64.promote_f32
- [x] f32.convert_i32_s
- [x] f32.convert_i64_s
- [x] f64.convert_i32_s
- [x] f64.convert_i64_s
- [x] f32.convert_i32_u
- [x] f32.convert_i64_u
- [x] f64.convert_i32_u
- [x] f64.convert_i64_u
- [x] i32.reinterpret_f32
- [x] i64.reinterpret_f64
- [x] f32.reinterpret_i32
- [x] f64.reinterpret_i64
- [x] i32.extend8_s
- [x] i32.extend16_s
- [x] i64.extend8_s
- [x] i64.extend16_s
- [x] i64.extend32_s

##### Load and Store Instructions
- [x] i32.load
- [x] i64.load
- [x] f32.load
- [x] f64.load
- [x] i32.store
- [x] i64.store
- [x] f32.store
- [x] f64.store
- [x] i32.load8_s
- [x] i32.load16_s
- [x] i64.load8_s
- [x] i64.load16_s
- [x] i64.load32_s
- [x] i32.load8_u
- [x] i32.load16_u
- [x] i64.load8_u
- [x] i64.load16_u
- [x] i64.load32_u
- [x] i32.store8
- [x] i32.store16
- [x] i64.store8
- [x] i64.store16
- [x] i64.store32

##### Memory Instructions
- [ ] memory.grow: Not needed due to context size limit and scope
- [x] memory.size: Currently always returns 1 (page), due to context size limit
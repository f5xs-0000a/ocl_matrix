use ocl::Buffer;
use core::cmp::min;
use core::cmp::max;
use crate::utils::bool32;
use ocl::prm::Uint4;
use ocl::ProQue;
use crate::utils::ForceTruncOrZeroRepeatIter;
use core::iter::repeat;
use core::iter::repeat_with;
use itertools::Itertools as _;

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    X,
    Y,
    Z,
    W,
}

impl Dimension {
    fn to_index(self) -> u32 {
        use self::Dimension::*;

        match self {
            X => 0,
            Y => 1,
            Z => 2,
            W => 3,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

lazy_static! {
    static ref PROQUE: ProQue = {
        let src = include_str!("opencl.cl");

        println!("Building OpenCL file...");
        let proque = ProQue::builder()
            .src(src)
            .dims(65536)
            .build()
            .unwrap();
        println!("OpenCL file built.");

        proque
    };
}

fn area(dims: &[u32; 4]) -> u32 {
    dims[0] * dims[1] * dims[2] * dims[3]
}

////////////////////////////////////////////////////////////////////////////////

pub struct Matrixf32 {
    xyzw:   [u32; 4],
    matrix: Buffer<f32>,
    meta:   Uint4,
}

impl Matrixf32 {
    pub fn dimensions<'a>(&'a self) -> &'a [u32; 4] {
        &self.xyzw
    }

    pub fn area(&self) -> u32 {
        area(&self.xyzw)
    }

    pub fn read(&self) -> Vec<(f32, [u32; 4])> {
        let mut storage = vec![0.; self.area() as usize];
        self.matrix.read(&mut storage).enq().unwrap();

        storage.into_iter()
            .scan([0, 0, 0, 0], |coords, val| {
                let retval = (val, coords.clone());

                for (coord, dim) in
                    coords.iter_mut().zip(self.dimensions().iter())
                {
                    *coord += 1;
                    if *coord >= *dim {
                        *coord = 0;
                    }
                    else {
                        break;
                    }
                }

                Some(retval)
            })
            .collect::<Vec<_>>()
    }

    pub fn new<I>(
        dims: [u32; 4],
        iter: I,
    ) -> Matrixf32
    where
        I: Iterator<Item = f32>,
    {
        let area_as_usize = area(&dims) as usize;
        let temp = ForceTruncOrZeroRepeatIter::new(iter, area_as_usize)
            .take(area_as_usize)
            .collect::<Vec<_>>();

        let buffer = PROQUE.buffer_builder::<f32>()
            .len(area_as_usize)
            .copy_host_slice(&*temp)
            .build()
            .unwrap();

        let meta = Uint4::new(dims[0], dims[1], dims[2], dims[3]);

        Matrixf32 {
            xyzw: dims,
            matrix: buffer,
            meta,
        }
    }

    pub fn new_fill(
        dims: [u32; 4],
        val: f32,
    ) -> Matrixf32{
        let area_as_usize = area(&dims) as usize;
        let buffer = PROQUE.buffer_builder::<f32>()
            .len(area_as_usize)
            .fill_val(val)
            .build()
            .unwrap();

        let meta = Uint4::new(dims[0], dims[1], dims[2], dims[3]);

        Matrixf32 {
            xyzw: dims,
            matrix: buffer,
            meta,
        }
    }

    pub fn copy(&self) -> Matrixf32 {
        let dest = Matrixf32::new(self.xyzw.clone(), repeat(0.));
        let area = self.area();

        let kernel = PROQUE.kernel_builder("copy")
            .arg(&self.matrix)
            .arg(area)
            .arg(&dest.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap(); }

        dest
    }

    pub fn zero(dims: [u32; 4]) -> Matrixf32 {
        Matrixf32::new(dims, repeat(0.))
    }

    pub fn identity(dims: [u32; 3]) -> Matrixf32 {
        let base_iter_gen = || {
            (0 .. dims[0])
                .cartesian_product(0 .. dims[0])
                .map(|(a, b)| a == b)
                .map(|tf| {
                    if tf {
                        1.
                    }
                    else {
                        0.
                    }
                })
        };
        let iter_gen = repeat_with(base_iter_gen).flat_map(|iter| iter);
        let dims = [dims[0], dims[0], dims[1], dims[2]];

        Matrixf32::new(dims, iter_gen)
    }

    pub fn add_eq(self, other: &Matrixf32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("add_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn mul_eq(self, other: &Matrixf32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("mul_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn sub_eq(self, other: &Matrixf32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("sub_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn div_eq(self, other: &Matrixf32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("div_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn matrix_mul_add_eq(self, left: &Matrixf32, right: &Matrixf32)
    -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("matrix_mul_add_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&left.matrix)
            .arg(&left.meta)
            .arg(&right.matrix)
            .arg(&right.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn sub_scalar_eq(
        self,
        val: f32
    ) -> Matrixf32 {
        // because of time constraints, we'll have to compromise for a bit.
        // we can optimize this, however. we just don't for now and settle for
        // less
        
        let dims = self.dimensions().clone();
        self.sub_eq(&Matrixf32::new_fill(dims, val))
    }

    pub fn mul_scalar_eq(
        self,
        val: f32
    ) -> Matrixf32 {
        // because of time constraints, we'll have to compromise for a bit.
        // we can optimize this, however. we just don't for now and settle for
        // less
        
        let dims = self.dimensions().clone();
        self.mul_eq(&Matrixf32::new_fill(dims, val))
    }

    pub fn add_scalar_eq(
        self,
        val: f32
    ) -> Matrixf32 {
        // because of time constraints, we'll have to compromise for a bit.
        // we can optimize this, however. we just don't for now and settle for
        // less
        
        let dims = self.dimensions().clone();
        self.add_eq(&Matrixf32::new_fill(dims, val))
    }

    pub fn div_scalar_eq(
        self,
        val: f32
    ) -> Matrixf32 {
        // because of time constraints, we'll have to compromise for a bit.
        // we can optimize this, however. we just don't for now and settle for
        // less
        
        let dims = self.dimensions().clone();
        self.div_eq(&Matrixf32::new_fill(dims, val))
    }

    pub fn matrix_mul(&self, right: &Matrixf32)
    -> Matrixf32 {
        let zero = Matrixf32::zero([
            right.dimensions()[0],
            self.dimensions()[1],
            min(self.dimensions()[2], right.dimensions()[2]),
            min(self.dimensions()[3], right.dimensions()[3]),
        ]);

        zero.matrix_mul_add_eq(self, right)
    }

    pub fn exp_eq(self) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("exp_eq")
            .arg(&self.matrix)
            .arg(self.area())
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn powf_scalar_eq(self, power: f32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("powf_eq")
            .arg(&self.matrix)
            .arg(self.area())
            .arg(power)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn powi_scalar_eq(self, power: i32) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("powi_eq")
            .arg(&self.matrix)
            .arg(self.area())
            .arg(power)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn sigmoid_eq(self) -> Matrixf32 {
        let kernel = PROQUE.kernel_builder("sigmoid_eq")
            .arg(&self.matrix)
            .arg(self.area())
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        self
    }

    pub fn flatten(&self, extend: Dimension, flatten: Dimension) -> Matrixf32 {
        let mut new_dims = self.xyzw.clone();
        new_dims[extend.to_index() as usize] *=
            new_dims[flatten.to_index() as usize];
        new_dims[flatten.to_index() as usize] = 1;

        let result = Matrixf32::zero(new_dims);

        let kernel = PROQUE.kernel_builder("flatten")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&result.matrix)
            .arg(extend.to_index())
            .arg(flatten.to_index())
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn extend(&self, other: &Matrixf32, direction :Dimension) -> Matrixf32 {
        let dim_index = direction.to_index() as usize;
        let mut result_dims = [
            max(self.xyzw[0], other.xyzw[0]),
            max(self.xyzw[1], other.xyzw[1]),
            max(self.xyzw[2], other.xyzw[2]),
            max(self.xyzw[3], other.xyzw[3]),
        ];
        result_dims[dim_index] =
            self.xyzw[dim_index] + other.xyzw[dim_index];

        let result = Matrixf32::zero(result_dims);

        let kernel = PROQUE.kernel_builder("extend")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .arg(&result.meta)
            .arg(direction.to_index())
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    fn retain_indices_inner(
        &self,
        indices: [Vec<u32>; 4],
    ) -> Matrixf32 {
        fn vec_to_buffer(vec: &[u32]) -> Buffer<u32> {
            PROQUE.buffer_builder::<u32>()
                .len(vec.len())
                .copy_host_slice(vec)
                .build()
                .unwrap()
        }

        let x_buffer = vec_to_buffer(&indices[0]);
        let y_buffer = vec_to_buffer(&indices[1]);
        let z_buffer = vec_to_buffer(&indices[2]);
        let w_buffer = vec_to_buffer(&indices[3]);

        let dest = Matrixf32::zero([
            indices[0].len() as u32,
            indices[1].len() as u32,
            indices[2].len() as u32,
            indices[3].len() as u32,
        ]);

        let kernel = PROQUE.kernel_builder("retain_indices")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&x_buffer)
            .arg(&y_buffer)
            .arg(&z_buffer)
            .arg(&w_buffer)
            .arg(&dest.matrix)
            .arg(&dest.meta)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        dest
    }

    /// Retains the indices in the matrix from the provided argument.
    ///
    /// This method does not retain indices that are outside of bounds.
    ///
    /// You can also retain an index multiple times. The said index will then
    /// appear multiple times.
    ///
    /// You can also reorder indices with this method.
    ///
    /// However, if you placed indices that are outside of bounds before an
    /// index that is within bounds, the index within bounds will take the place
    /// of the index that is outside of bounds, returning you with a matrix with
    /// an n-area less than expected.
    pub fn retain_indices(
        &self,
        mut xyzw: [Option<Vec<u32>>; 4],
    ) -> Matrixf32 {
        let mut retain_indices = <[Vec<u32>; 4]>::default();

        for idx in 0 .. 4 {
            if let Some(mut dimvec) = xyzw[idx].take() {
                dimvec.retain(|x| *x < self.xyzw[idx]);
                retain_indices[idx] = dimvec;
            }
            else {
                retain_indices[idx] = (0 .. self.xyzw[idx]).collect();
            }
        }

        self.retain_indices_inner(retain_indices)
    }

    pub fn remove_indices(
        self,
        mut indices: [Vec<u32>; 4],
    ) -> Matrixf32
    {
        let mut retain_indices = <[Vec<u32>; 4]>::default();

        for idx in 0 .. 4 {
            indices[idx].retain(|x| *x < self.xyzw[idx]);
            indices[idx].sort_unstable();
            indices[idx].dedup();

            retain_indices[idx] = (0 .. self.xyzw[idx])
                // TODO: you know, iterating through both of them would probably
                // be faster but, you know, I'm in a hurry. premature
                // optimization shouldn't be my thing
                .filter(|i| indices[idx].binary_search(i).is_err())
                .collect();
        }

        self.retain_indices_inner(retain_indices)
    }

    pub fn repeat_along(
        &self,
        repeat: [u32; 4],
    ) -> Matrixf32
    {
        // TODO: in the future, we should create a dedicated shader for this
        // since that is more efficient. However, we have time restraints so
        // we make do with what we have.

        let mut retain_indices = <[Vec<u32>; 4]>::default();

        for idx in 0 .. 4 {
            retain_indices[idx] = repeat_with(|| (0 .. self.xyzw[idx]))
                .take(repeat[idx] as usize)
                .flat_map(|iter| iter)
                .collect();
        }

        self.retain_indices_inner(retain_indices)
    }

    pub fn transpose(
        &self,
        dim1: Dimension,
        dim2: Dimension,
    ) -> Matrixf32
    {
        let mut result_dims = self.xyzw.clone();
        result_dims.swap(dim1.to_index() as usize, dim2.to_index() as usize);

        let result = Matrixf32::zero(result_dims);

        let kernel = PROQUE.kernel_builder("transpose")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&result.matrix)
            .arg(dim1.to_index())
            .arg(dim2.to_index())
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn gt_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_gt")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn ge_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_ge")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn lt_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_lt")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn le_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_le")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn eq_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn ne_matrix(&self, other: &Self) -> MatrixBool {
        let result_dims = [
            min(self.dimensions()[0], other.dimensions()[0]),
            min(self.dimensions()[1], other.dimensions()[1]),
            min(self.dimensions()[2], other.dimensions()[2]),
            min(self.dimensions()[3], other.dimensions()[3]),
        ];

        let result = MatrixBool::f(result_dims);

        let kernel = PROQUE.kernel_builder("f32_ne")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn is_nan(&self) -> MatrixBool {
        let result = MatrixBool::f(self.dimensions().clone());

        let kernel = PROQUE.kernel_builder("is_nan")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn if_else(&self, cond: &MatrixBool, other: &Matrixf32) -> Matrixf32 {
        let result_dims = [
            min(min(self.xyzw[0], cond.xyzw[0]), other.xyzw[0]),
            min(min(self.xyzw[0], cond.xyzw[0]), other.xyzw[1]),
            min(min(self.xyzw[0], cond.xyzw[0]), other.xyzw[2]),
            min(min(self.xyzw[0], cond.xyzw[0]), other.xyzw[3]),
        ];
        let result = Matrixf32::zero(result_dims);

        let kernel = PROQUE.kernel_builder("if_else_eq")
            .arg(&self.matrix)
            .arg(&self.meta)
            .arg(&other.matrix)
            .arg(&other.meta)
            .arg(&cond.matrix)
            .arg(&cond.meta)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        result
    }

    pub fn sum_along(&self, dimensions: [bool; 4]) -> Matrixf32 {
        // copy the contents of the buffer
        let mut last_vec =
            self.read().iter().map(|(val, _)| val).cloned().collect::<Vec<_>>();
        let mut new_dims = self.xyzw.clone();

        fn get_area(dims: &[u32; 4]) -> u32 {
            dims[0] * dims[1] * dims[2] * dims[3]
        }

        fn get_index(
            coords: [u32; 4],
            dims: &[u32; 4],
        ) -> usize
        {
            (coords[3] * dims[0] * dims[1] * dims[2]
                + coords[2] * dims[0] * dims[1]
                + coords[1] * dims[0]
                + coords[0]) as usize
        }

        fn get_coords(
            mut index: u32,
            dims: &[u32; 4],
        ) -> [u32; 4]
        {
            let mut result = [0; 4];

            result[3] = index / (dims[0] * dims[1] * dims[2]);
            index %= dims[0] * dims[1] * dims[2];

            result[2] = index / (dims[0] * dims[1]);
            index %= dims[0] * dims[1];

            result[1] = index / dims[0];
            index %= dims[0];

            result[0] = index;

            result
        }

        // iterate through each of the elements, summing up which of the
        // dimensions that need to be summed up and replacing the last vec and
        // new dims in the process
        dimensions.iter().enumerate().filter(|(_, flag)| **flag).for_each(
            |(side, _)| {
                use core::cmp::min;

                let count = new_dims[side].clone();
                let old_dims = new_dims.clone();

                // set the new dims to 1
                new_dims[side] = min(new_dims[side], 1);

                // determine the new n-area;
                let new_length = get_area(&new_dims);

                last_vec = (0 .. new_length)
                    .map(|idx| {
                        last_vec
                            .iter()
                            .cloned()
                            .skip({
                                // get the coords of an index using the area of
                                // the destination
                                let coords = get_coords(idx, &new_dims);

                                // convert that coords to an index using the
                                // area
                                // of the source
                                get_index(coords, &old_dims)
                            })
                            .step_by({
                                let x_i = get_index([0; 4], &old_dims);
                                let x_f = {
                                    let mut f = [0; 4];
                                    f[side] = 1;
                                    get_index(f, &old_dims)
                                };
                                x_f - x_i
                            })
                            .take(count as usize)
                            .sum::<f32>()
                    })
                    .collect();
            },
        );

        let buffer = PROQUE.buffer_builder::<f32>()
            .len(last_vec.len())
            .copy_host_slice(&*last_vec)
            .build()
            .unwrap();

        let meta = Uint4::new(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);

        Matrixf32 {
            xyzw: new_dims,
            matrix: buffer,
            meta,
        }
    }

    /*
    pub fn sum_along(&self, dimensions: [bool; 4]) -> Matrixf32 {
        fn reducer<F>(
            source: &[u32; 4],
            dimensions: &[bool; 4],
            reducer: F
        ) -> [u32; 4]
        where F: Fn(u32) -> u32 {
            let mut reduced = [0u32; 4];
            reduced.iter_mut()
                .zip(source.iter())
                .zip(dimensions.iter())
                .map(|((dim, former), xyzw)| (dim, former, xyzw))
                .for_each(|(dim, former, xyzw)| {
                    *dim = if *xyzw {
                        reducer(*former)
                    } else {
                        *former
                    }
                });

            reduced
        }

        let result = Matrixf32::zero(
            reducer(
                &self.xyzw,
                &dimensions,
                |_| 1,
            )
        );

        let dim32 = [
            bool32::from(dimensions[0]),
            bool32::from(dimensions[1]),
            bool32::from(dimensions[2]),
            bool32::from(dimensions[3]),
        ];

        let sum_flags = PROQUE.buffer_builder::<bool32>()
            .len(4)
            .copy_host_slice(&dim32)
            .build()
            .unwrap();

        let intermediate_dims = reducer(
            &self.xyzw,
            &dimensions,
            |x| ((x + 1) / 2) * 2,
        );

        let intermediate_area = area(&intermediate_dims);

        let intermediate = PROQUE.buffer_builder::<f32>()
            .len(intermediate_area)
            .fill_val(0.)
            .build()
            .unwrap();

        let intermediate_meta = Uint4::new(
            intermediate_dims[0],
            intermediate_dims[1],
            intermediate_dims[2],
            intermediate_dims[3],
        );

        let kernel = PROQUE.kernel_builder("sum_along")
            .arg(&self.matrix)
            .arg(&intermediate_meta)
            .arg(&sum_flags)
            .arg(&intermediate)
            .arg(&result.matrix)
            .build()
            .unwrap();

        unsafe { kernel.enq().unwrap() }

        let mut storage = vec![0.; intermediate_area as usize];
        intermediate.read(&mut storage).enq().unwrap();
        dbg!(storage);

        result
    }
    */
}

////////////////////////////////////////////////////////////////////////////////

pub struct MatrixBool {
    xyzw:   [u32; 4],
    matrix: Buffer<bool32>,
    meta:   Uint4,
}

impl MatrixBool {
    pub fn dimensions<'a>(&'a self) -> &'a [u32; 4] {
        &self.xyzw
    }

    pub fn area(&self) -> u32 {
        area(&self.xyzw)
    }

    pub fn read(&self) -> Vec<(bool, [u32; 4])> {
        let mut storage = vec![bool32::f(); self.area() as usize];
        self.matrix.read(&mut storage).enq().unwrap();

        storage.into_iter()
            .scan([0, 0, 0, 0], |coords, val| {
                let retval = (val.into(), coords.clone());

                for (coord, dim) in
                    coords.iter_mut().zip(self.dimensions().iter())
                {
                    *coord += 1;
                    if *coord >= *dim {
                        *coord = 0;
                    }
                    else {
                        break;
                    }
                }

                Some(retval)
            })
            .collect::<Vec<_>>()
    }

    pub fn new<I>(
        dims: [u32; 4],
        iter: I,
    ) -> MatrixBool
    where
        I: Iterator<Item = bool32>,
    {
        let area_as_usize = area(&dims) as usize;
        let temp = ForceTruncOrZeroRepeatIter::new(iter, area_as_usize)
            .take(area_as_usize)
            .collect::<Vec<_>>();

        let buffer = PROQUE.buffer_builder::<bool32>()
            .len(area_as_usize)
            .copy_host_slice(&*temp)
            .build()
            .unwrap();

        let meta = Uint4::new(dims[0], dims[1], dims[2], dims[3]);

        MatrixBool {
            xyzw: dims,
            matrix: buffer,
            meta,
        }
    }

    pub fn new_fill(
        dims: [u32; 4],
        val: bool,
    ) -> MatrixBool {
        let area_as_usize = area(&dims) as usize;
        let buffer = PROQUE.buffer_builder::<bool32>()
            .len(area_as_usize)
            .fill_val(bool32::from(val))
            .build()
            .unwrap();

        let meta = Uint4::new(dims[0], dims[1], dims[2], dims[3]);

        MatrixBool {
            xyzw: dims,
            matrix: buffer,
            meta,
        }
    }

    pub fn f(dims: [u32; 4]) -> MatrixBool {
        MatrixBool::new(dims, repeat(bool32::f()))
    }
}
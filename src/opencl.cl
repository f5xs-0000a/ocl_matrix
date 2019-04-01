void get_workable_mem_and_offset(
    uint len,
    uint* invoc_workable_mem,
    uint* invoc_offset
) {
    uint invoc_index = get_global_id(0);
    uint invoc_count = get_global_size(0);

    *invoc_workable_mem = (len - 1) / invoc_count + 1;
    *invoc_offset = invoc_index * *invoc_workable_mem;
}

uint get_area(
    uint4* dims
) {
    return (*dims)[0] * (*dims)[1] * (*dims)[2] * (*dims)[3];
}

uint get_index(
    uint4* coords,
    uint4* dims
) {
    return
        (*coords)[3] * (*dims)[0] * (*dims)[1] * (*dims)[2] +
        (*coords)[2] * (*dims)[0] * (*dims)[1] +
        (*coords)[1] * (*dims)[0] +
        (*coords)[0];
}

uint4 get_coords(
    uint index,
    uint4* dims
) {
    uint4 result;

    {
        uint denom = (*dims)[0] * (*dims)[1] * (*dims)[2];
        result[3] = index / denom;
        index %= denom;
    }

    {
        uint denom = (*dims)[0] * (*dims)[1];
        result[2] = index / denom;
        index %= denom;
    }

    {
        uint denom = (*dims)[0];
        result[1] = index / denom;
        index %= denom;
    }

    result[0] = index;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

uint4 min_dims(
    uint4* left_dims,
    uint4* right_dims
) {
    uint4 new_dims;

    new_dims[0] = min((*left_dims)[0], (*right_dims)[0]);
    new_dims[1] = min((*left_dims)[1], (*right_dims)[1]);
    new_dims[2] = min((*left_dims)[2], (*right_dims)[2]);
    new_dims[3] = min((*left_dims)[3], (*right_dims)[3]);

    return new_dims;
}

uint4 matrix_mul_common_dims_and_mn_out(
    uint4* left,
    uint4* right,
    uint4* product,
    uint* mn_out
) {
    uint4 new_dims;

    new_dims[0] = min((*right)[0], (*product)[0]);
    new_dims[1] = min((*left)[1], (*product)[1]);
    new_dims[2] = min(min((*left)[2], (*right)[2]), (*product)[2]);
    new_dims[3] = min(min((*left)[3], (*right)[3]), (*product)[3]);

    *mn_out = min((*left)[0], (*right)[1]);

    return new_dims;
}

////////////////////////////////////////////////////////////////////////////////

kernel void
copy(
    global float* src_buffer,
    uint area,
    global float* dest_buffer
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        dest_buffer[cur_invoc_index] = src_buffer[cur_invoc_index];
    }
}

kernel void
add_eq(
    global float* opeq_buffer,
    uint4 opeq_meta,
    global float* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] += opand_buffer[opand_index];
        }
    }
}

kernel void
sub_eq(
    global float* opeq_buffer,
    uint4 opeq_meta,
    global float* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] -= opand_buffer[opand_index];
        }
    }
}

kernel void
mul_eq(
    global float* opeq_buffer,
    uint4 opeq_meta,
    global float* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] *= opand_buffer[opand_index];
        }
    }
}

kernel void
div_eq(
    global float* opeq_buffer,
    uint4 opeq_meta,
    global float* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] /= opand_buffer[opand_index];
        }
    }
}

kernel void
matrix_mul_add_eq(
    global float* product_buffer,
    uint4 product_meta,
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta
) {
    uint mn_out;
    uint4 common_dims = matrix_mul_common_dims_and_mn_out(
        &left_meta,
        &right_meta,
        &product_meta,
        &mn_out
    );
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 cur_coords = get_coords(cur_invoc_index, &common_dims);

            uint prod_index = get_index(&cur_coords, &product_meta);
            float answer = 0;

            for (uint cur_xy = 0; cur_xy < mn_out; cur_xy += 1) {
                uint4 left_coords = cur_coords;
                left_coords[0] = cur_xy;
                uint left_index = get_index(&left_coords, &left_meta);

                uint4 right_coords = cur_coords;
                right_coords[1] = cur_xy;
                uint right_index = get_index(&right_coords, &right_meta);

                answer += left_buffer[left_index] * right_buffer[right_index];
            }

            product_buffer[prod_index] += answer;
        }
    }
}

kernel void
exp_eq(
    global float* src_buffer,
    uint area
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            src_buffer[cur_invoc_index] = exp(src_buffer[cur_invoc_index]);
        }
    }
}

kernel void
powf_eq(
    global float* src_buffer,
    uint area,
    float power
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            src_buffer[cur_invoc_index] =
                pow(src_buffer[cur_invoc_index], power);
        }
    }
}

kernel void
powi_eq(
    global float* src_buffer,
    uint area,
    int power
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            src_buffer[cur_invoc_index] =
                pow(src_buffer[cur_invoc_index], power);
        }
    }
}

kernel void
sigmoid_eq(
    global float* src_buffer,
    uint area
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            float x_exp = exp(src_buffer[cur_invoc_index]);
            src_buffer[cur_invoc_index] = x_exp / (x_exp + 1.);
        }
    }
}

kernel void flatten(
    global float* src_buffer,
    uint4 src_meta,
    global float* dest_buffer,
    uint extend_dim,
    uint flatten_dim
) {
    uint array_len = get_area(&src_meta);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    // declare the dimensions of the destination marix
    uint4 dest_dim = src_meta; {
        // extend the extended dimension by multiplying it by the flattened
        // dimension
        dest_dim[extend_dim] *= dest_dim[flatten_dim];

        // flatten the flattened dimension by setting it to 1
        dest_dim[flatten_dim] = 1;
    }

    // perform copy from base to result
    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        // get the current inex
        uint src_index = (invoc_offset + invoc_loop_index) % array_len;

        // get the coordinates given the index
        // get the index on the destination
        uint4 coords = get_coords(src_index, &src_meta);

        // extend the extended dimension
        coords[extend_dim] += coords[flatten_dim] * src_meta[extend_dim];

        // flatten the flattened dimension
        coords[flatten_dim] = 0;

        uint dest_index = get_index(&coords, &dest_dim);

        dest_buffer[dest_index] = src_buffer[src_index];
    }
}

kernel void extend(
    global float* src_buffer,
    uint4 src_meta,
    global float* ext_buffer,
    uint4 ext_meta,
    global float* result_buffer,
    uint4 result_meta,
    uint extension_direction
) {
    { // copy the first buffer
        uint array_length = get_area(&src_meta);
        uint invoc_workable_mem;
        uint invoc_offset;
        get_workable_mem_and_offset(
            array_length,
            &invoc_workable_mem,
            &invoc_offset
        );

        // perform copy from base to result
        for (
             uint invoc_loop_index = 0;
             invoc_loop_index < invoc_workable_mem;
             invoc_loop_index += 1
        ) {
            // get the current index
            uint src_index = (invoc_offset + invoc_loop_index) % array_length;

            // get the index on the destination
            uint4 coords = get_coords(src_index, &src_meta);
            uint dest_index = get_index(&coords, &result_meta);

            result_buffer[dest_index] = src_buffer[src_index];
        }
    }

    { // copy the second buffer
        uint array_length = get_area(&ext_meta);
        uint invoc_workable_mem;
        uint invoc_offset;
        get_workable_mem_and_offset(
            array_length,
            &invoc_workable_mem,
            &invoc_offset
        );

        // perform copy from base to result
        for (
             uint invoc_loop_index = 0;
             invoc_loop_index < invoc_workable_mem;
             invoc_loop_index += 1
        ) {
            // get the current index
            uint src_index = (invoc_offset + invoc_loop_index) % array_length;

            // get the index on the destination
            uint4 dest_coords = get_coords(src_index, &ext_meta);
            dest_coords[extension_direction] += src_meta[extension_direction];
            uint dest_index = get_index(&dest_coords, &result_meta);

            result_buffer[dest_index] = ext_buffer[src_index];
        }
    }
}

kernel void retain_indices(
    global float* src_buffer,
    uint4 src_meta,
    global uint* x_indices,
    global uint* y_indices,
    global uint* z_indices,
    global uint* w_indices,
    global float* dest_buffer,
    uint4 dest_meta
) {
    uint array_len = get_area(&dest_meta);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    // perform copy from base to result
    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        // get the current inex
        uint dest_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 dest_coords = get_coords(dest_index, &dest_meta);

        uint4 src_coords;
        src_coords[0] = x_indices[dest_coords[0]];
        src_coords[1] = y_indices[dest_coords[1]];
        src_coords[2] = z_indices[dest_coords[2]];
        src_coords[3] = w_indices[dest_coords[3]];

        uint src_index = get_index(&src_coords, &src_meta);
        dest_buffer[dest_index] = src_buffer[src_index];
    }
}

kernel void transpose(
    global float* src_buffer,
    uint4 src_meta,
    global float* dest_buffer,
    uint to_dim,
    uint from_dim
) {
    uint array_len = get_area(&src_meta);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    // declare the dimensions of the destination matrix
    uint4 dest_dim = src_meta; {
        uint temp = dest_dim[to_dim];
        dest_dim[to_dim] = dest_dim[from_dim];
        dest_dim[from_dim] = temp;
    }

    // perform copy from base to result
    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        // get the current index
        uint src_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(src_index, &src_meta);

        // swap
        uint temp = coords[to_dim];
        coords[to_dim] = coords[from_dim];
        coords[from_dim] = temp;

        uint dest_index = get_index(&coords, &dest_dim);
        dest_buffer[dest_index] = src_buffer[src_index];
    }
}

kernel void
f32_eq(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] == right_buffer[right_index];
    }
}

kernel void
f32_ne(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] != right_buffer[right_index];
    }
}

kernel void
f32_gt(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] > right_buffer[right_index];
    }
}

kernel void
f32_ge(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] >= right_buffer[right_index];
    }
}

kernel void
f32_lt(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] < right_buffer[right_index];
    }
}

kernel void
f32_le(
    global float* left_buffer,
    uint4 left_meta,
    global float* right_buffer,
    uint4 right_meta,
    global bool* result_buffer
) {
    uint4 common_dims = min_dims(&left_meta, &right_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 coords = get_coords(cur_invoc_index, &common_dims);
        uint left_index = get_index(&coords, &left_meta);
        uint right_index = get_index(&coords, &right_meta);

        result_buffer[cur_invoc_index] =
            left_buffer[left_index] <= right_buffer[right_index];
    }
}

kernel void
is_nan(
    global float* left_buffer,
    uint area,
    global bool* result_buffer
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        result_buffer[cur_invoc_index] = isnan(left_buffer[cur_invoc_index]);
    }
}

kernel void
if_else_eq(
    global float* if_buffer,
    uint4 if_meta,
    global float* else_buffer,
    uint4 else_meta,
    global bool* cond_buffer,
    uint4 cond_meta,
    global float* result_buffer
) {
    uint4 pre_dims = min_dims(&if_meta, &else_meta);
    uint4 common_dims = min_dims(&pre_dims, &cond_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
        uint invoc_loop_index = 0;
        invoc_loop_index < invoc_workable_mem;
        invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_len;
        uint4 cur_coords = get_coords(cur_invoc_index, &common_dims);

        uint if_index = get_index(&cur_coords, &if_meta);
        uint else_index = get_index(&cur_coords, &else_meta);
        uint cond_index = get_index(&cur_coords, &cond_meta);

        if (cond_buffer[cond_index]) {
            result_buffer[cur_invoc_index] = if_buffer[if_index];
        }

        else {
            result_buffer[cur_invoc_index] = else_buffer[else_index];
        }
    }
}

kernel void
sum_along(
    global float* src_buffer,
    uint4 src_meta,
    uint sum_direction,
    global float* dest_buffer
) {
    uint4 dest_dims = src_meta;
    dest_dims[sum_direction] = 1;

    uint array_length = get_area(&dest_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_length,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_length;
        uint4 cur_coords = get_coords(cur_invoc_index, &dest_dims);

        float sum = 0;

        for (
             uint i = 0;
             i < src_meta[sum_direction];
             i += 1
        ) {
            uint4 coords = cur_coords;
            coords[sum_direction] = i;
            uint index = get_index(&coords, &src_meta);

            sum += src_buffer[index];
        }

        dest_buffer[cur_invoc_index] = sum;
    }
}

kernel void
max_along(
    global float* src_buffer,
    uint4 src_meta,
    uint max_direction,
    global float* dest_buffer
) {
    uint4 dest_dims = src_meta;
    dest_dims[max_direction] = 1;

    uint array_length = get_area(&dest_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_length,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_length;
        uint4 cur_coords = get_coords(cur_invoc_index, &dest_dims);

        float maximum = -MAXFLOAT;

        for (
             uint i = 0;
             i < src_meta[max_direction];
             i += 1
        ) {
            uint4 coords = cur_coords;
            coords[max_direction] = i;
            uint index = get_index(&coords, &src_meta);

            maximum = max(src_buffer[index], maximum);
        }

        dest_buffer[cur_invoc_index] = maximum;
    }
}

kernel void
min_along(
    global float* src_buffer,
    uint4 src_meta,
    uint min_direction,
    global float* dest_buffer
) {
    uint4 dest_dims = src_meta;
    dest_dims[min_direction] = 1;

    uint array_length = get_area(&dest_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_length,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = (invoc_offset + invoc_loop_index) % array_length;
        uint4 cur_coords = get_coords(cur_invoc_index, &dest_dims);

        float minimum = MAXFLOAT;

        for (
             uint i = 0;
             i < src_meta[min_direction];
             i += 1
        ) {
            uint4 coords = cur_coords;
            coords[min_direction] = i;
            uint index = get_index(&coords, &src_meta);

            minimum = min(src_buffer[index], minimum);
        }

        dest_buffer[cur_invoc_index] = minimum;
    }
}

/*
kernel void
sum_along(
    global float* src_buffer,
    uint4 intermediate_meta,
    global bool sum_flags[4],
    global float* intermediate,
    global float* dest_buffer
) {
    uint4 src_mat = intermediate_meta;

    ////////  at this point, we just copy the input matrix to the intermediate1
    ////////  matrix  //////////////////////////////////////////////////////////

    {
        uint array_length = get_area(&src_mat);
        uint invoc_workable_mem;
        uint invoc_offset;
        get_workable_mem_and_offset(
            array_length,
            &invoc_workable_mem,
            &invoc_offset
        );

        for (
            uint invoc_loop_index = 0;
            invoc_loop_index < invoc_workable_mem;
            invoc_loop_index += 1
        ) {
            uint cur_invoc_index =
                (invoc_offset + invoc_loop_index) % array_length;

            intermediate[cur_invoc_index] =
                src_buffer[cur_invoc_index];
        }
    }

    //////// then we perform the shuffling between the two intermediates  //////

    // set the flag to determine to which buffer we will write into
    // this should be initialized to false since we have already copied the data
    // from the input to the first buffer, so therefore we must write into
    // the second buffer
    // iterate for each of the components

    for (uint comp = 0; comp < 4; comp += 1) {
        
    // if the flag is available
    if (sum_flags[comp]) {

    // while the length of the dimension being summed is still
    // greater than 1
    while (src_mat[comp] > 1) {
        uint4 dest_mat = src_mat;
        dest_mat[comp] /= 2;

        uint array_length = get_area(&dest_mat);
        uint invoc_workable_mem;
        uint invoc_offset;
        get_workable_mem_and_offset(
            array_length,
            &invoc_workable_mem,
            &invoc_offset
        );

        for (
            uint invoc_loop_index = 0;
            invoc_loop_index < invoc_workable_mem;
            invoc_loop_index += 1
        ) {
            uint dest_index = (invoc_offset + invoc_loop_index) % array_length;
            uint4 dest_coords = get_coords(dest_index, &dest_mat);

            uint4 src_coords_1 = dest_coords;
            src_coords_1[comp] *= 2;
            uint4 src_coords_2 = src_coords_1;
            src_coords_2[comp] += 1;

            float sum = 0;

            {
                uint src_index = get_index(&src_coords_1, &src_mat);
                sum += intermediate[src_index];
            }

            if (src_coords_2[comp] < src_mat[comp]) {
                uint src_index = get_index(&src_coords_2, &src_mat);
                sum += intermediate[src_index];
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
            intermediate[dest_index] = sum;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        // set the source matrix to the destination matrix
        src_mat = dest_mat;
        return;
    }
    }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    ////////  then we copy from either of the intermediates to the destination
    ////////  buffer  //////////////////////////////////////////////////////////

    {
        uint new_area = get_area(&src_mat);
        uint invoc_workable_mem;
        uint invoc_offset;
        get_workable_mem_and_offset(
            new_area,
            &invoc_workable_mem,
            &invoc_offset
        );

        for (
            uint invoc_loop_index = 0;
            invoc_loop_index < invoc_workable_mem;
            invoc_loop_index += 1
        ) {
            uint index = (invoc_offset + invoc_loop_index) % new_area;

            dest_buffer[index] = intermediate[index];
        }
    }
}
*/

////////////////////////////////////////////////////////////////////////////////

kernel void
and_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                opeq_buffer[opeq_index] && opand_buffer[opand_index];
        }
    }
}

kernel void
or_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                opeq_buffer[opeq_index] || opand_buffer[opand_index];
        }
    }
}

kernel void
xor_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                opeq_buffer[opeq_index] != opand_buffer[opand_index];
        }
    }
}

kernel void
nand_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                !(opeq_buffer[opeq_index] && opand_buffer[opand_index]);
        }
    }
}

kernel void
nor_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                !(opeq_buffer[opeq_index] || opand_buffer[opand_index]);
        }
    }
}

kernel void
xnor_eq(
    global bool* opeq_buffer,
    uint4 opeq_meta,
    global bool* opand_buffer,
    uint4 opand_meta
) {
    uint4 common_dims = min_dims(&opeq_meta, &opand_meta);
    uint array_len = get_area(&common_dims);
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            uint4 coords = get_coords(cur_invoc_index, &common_dims);
            uint opeq_index = get_index(&coords, &opeq_meta);
            uint opand_index = get_index(&coords, &opand_meta);

            opeq_buffer[opeq_index] =
                opeq_buffer[opeq_index] == opand_buffer[opand_index];
        }
    }
}

kernel void
not_eq(
    global bool* src_buffer,
    uint area
) {
    uint array_len = area;
    uint invoc_workable_mem;
    uint invoc_offset;
    get_workable_mem_and_offset(
        array_len,
        &invoc_workable_mem,
        &invoc_offset
    );

    for (
         uint invoc_loop_index = 0;
         invoc_loop_index < invoc_workable_mem;
         invoc_loop_index += 1
    ) {
        uint cur_invoc_index = invoc_offset + invoc_loop_index;
        if (cur_invoc_index < array_len) {
            src_buffer[cur_invoc_index] = !src_buffer[cur_invoc_index];
        }
    }
}

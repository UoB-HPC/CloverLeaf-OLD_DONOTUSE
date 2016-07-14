#include "definitions_c.h"
#include <stdlib.h>
#include "initialise_chunk.h"
#include "generate_chunk.h"
#include "ideal_gas.h"
#include "update_halo.h"
#include "field_summary.h"
#include "visit.h"
#include "clover.h"
#include "allocate.h"

void clover_decompose(int x_cells,
                      int y_cells,
                      int* left,
                      int* right,
                      int* bottom,
                      int* top);
void clover_tile_decompose(int, int);
void clover_allocate_buffers();

void start()
{
    BOSSPRINT(g_out, "\nSetting up initial geometry\n");
    _time = 0.0;
    step = 0;
    dtold = dtinit;
    dt = dtinit;

    clover_barrier();

    clover_get_num_chunks(&number_of_chunks);

    BOSSPRINT(g_out, "Number of chunks: %d\n", number_of_chunks);
    int left, right, bottom, top;
    clover_decompose(grid.x_cells,
                     grid.y_cells,
                     &left,
                     &right,
                     &bottom,
                     &top);


    chunk.task = parallel.task;
    int x_cells = right - left + 1;
    int y_cells = top - bottom + 1;

    chunk.left = left;
    chunk.bottom = bottom;
    chunk.right = right;
    chunk.top = top;
    chunk.left_boundary = 1;
    chunk.bottom_boundary = 1;
    chunk.right_boundary = grid.x_cells;
    chunk.top_boundary = grid.y_cells;
    chunk.x_min = 1;
    chunk.y_min = 1;
    chunk.x_max = x_cells;
    chunk.y_max = y_cells;

    chunk.tiles = (struct tile_type*)malloc(sizeof(struct tile_type) * tiles_per_chunk);

    clover_tile_decompose(x_cells, y_cells);

    allocate();
    clover_allocate_buffers();

    BOSSPRINT(g_out, "Generating chunks\n");

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        initialise_chunk(tile);
        generate_chunk(tile);
    }
    advect_x = true;

    clover_barrier();

    bool profiler_off = profiler_on;
    profiler_on = false;

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        ideal_gas(tile, false);
    }

    int fields[NUM_FIELDS];

    for (int i = 0; i < NUM_FIELDS; i++) {
        fields[i] = 0;
    }

    fields[FIELD_DENSITY0] = 1;
    fields[FIELD_ENERGY0] = 1;
    fields[FIELD_PRESSURE] = 1;
    fields[FIELD_VISCOSITY] = 1;
    fields[FIELD_DENSITY1] = 1;
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_XVEL0] = 1;
    fields[FIELD_YVEL0] = 1;
    fields[FIELD_XVEL1] = 1;
    fields[FIELD_YVEL1] = 1;


    update_halo(fields, 2);

    BOSSPRINT(g_out, "Problem initialised and generated\n");

    field_summary();
    clover_barrier();

    if (visit_frequency != 0) visit();

    profiler_on = profiler_off;
}

void clover_allocate_buffers()
{
    if (parallel.task == chunk.task) {
        // !IF(chunk.chunk_neighbours(chunk_left).NE.external_face) THEN
        chunk.left_snd_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.y_max + 5)));
        chunk.left_rcv_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.y_max + 5)));
        // !ENDIF
        // !IF(chunk.chunk_neighbours(chunk_right).NE.external_face) THEN
        chunk.right_snd_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.y_max + 5)));
        chunk.right_rcv_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.y_max + 5)));
        // !ENDIF
        // !IF(chunk.chunk_neighbours(chunk_bottom).NE.external_face) THEN
        chunk.bottom_snd_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.x_max + 5)));
        chunk.bottom_rcv_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.x_max + 5)));
        // !ENDIF
        // !IF(chunk.chunk_neighbours(chunk_top).NE.external_face) THEN
        chunk.top_snd_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.x_max + 5)));
        chunk.top_rcv_buffer = (double*)calloc(sizeof(double), (10 * 2 * (chunk.x_max + 5)));
        // !ENDIF
    }
}

void setArrayToVal(int* arr, int size, int val)
{
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

void clover_tile_decompose(int chunk_x_cells, int chunk_y_cells)
{
    int chunk_mesh_ratio,
        tile_x, tile_y,
        split_found,
        factor_x, factor_y,
        t,
        chunk_delta_x, chunk_delta_y,
        chunk_mod_x, chunk_mod_y,
        add_x_prev, add_y_prev,
        tile, tx, ty,
        add_x, add_y,
        left, right, top, bottom;

    chunk_mesh_ratio = (double)chunk_x_cells / chunk_y_cells;
    tile_x = tiles_per_chunk;
    tile_y = 1;

    split_found = 0;

    for (t = 1; t <= tiles_per_chunk; t++) {
        if (tiles_per_chunk % t == 0) {
            factor_x = tiles_per_chunk / (double)t;
            factor_y = t;

            if (factor_x / factor_y <= chunk_mesh_ratio) {
                tile_y = t;
                tile_x = tiles_per_chunk / t;
                split_found = 1;
            }
        }
    }

    if (split_found == 0 || tile_y == tiles_per_chunk) {
        if (chunk_mesh_ratio >= 1.0) {
            tile_x = tiles_per_chunk;
            tile_y = 1;
        } else {
            tile_x = 1;
            tile_y = tiles_per_chunk;
        }
    }

    chunk_delta_x = chunk_x_cells / tile_x;
    chunk_delta_y = chunk_y_cells / tile_y;
    chunk_mod_x = chunk_x_cells % tile_x;
    chunk_mod_y = chunk_y_cells % tile_y;

    add_x_prev = 0;
    add_y_prev = 0;

    tile = 0;
    for (ty = 1; ty <= tile_y; ty++) {
        for (tx = 1; tx <= tile_x; tx++) {
            add_x = 0;
            add_y = 0;
            if (tx <= chunk_mod_x) add_x = 1;
            if (ty <= chunk_mod_y) add_y = 1;

            left = chunk.left + (tx - 1) * chunk_delta_x + add_x_prev;
            right = left + chunk_delta_x - 1 + add_x;
            bottom = chunk.bottom + (ty - 1) * chunk_delta_y + add_y_prev;
            top = bottom + chunk_delta_y - 1 + add_y;

            chunk.tiles[tile].tile_neighbours[TILE_LEFT] = tile_x * (ty - 1) + tx - 1;
            chunk.tiles[tile].tile_neighbours[TILE_RIGHT] = tile_x * (ty - 1) + tx + 1;
            chunk.tiles[tile].tile_neighbours[TILE_BOTTOM] = tile_x * (ty - 2) + tx;
            chunk.tiles[tile].tile_neighbours[TILE_TOP] = tile_x * ty + tx;

            setArrayToVal(chunk.tiles[tile].external_tile_mask, 4, 0);

            if (tx == 1) {
                chunk.tiles[tile].tile_neighbours[TILE_LEFT] = EXTERNAL_TILE;
                chunk.tiles[tile].external_tile_mask[TILE_LEFT] = 1;
            }
            if (tx == tile_x) {
                chunk.tiles[tile].tile_neighbours[TILE_RIGHT] = EXTERNAL_TILE;
                chunk.tiles[tile].external_tile_mask[TILE_RIGHT] = 1;
            }
            if (ty == 1) {
                chunk.tiles[tile].tile_neighbours[TILE_BOTTOM] = EXTERNAL_TILE;
                chunk.tiles[tile].external_tile_mask[TILE_BOTTOM] = 1;
            }
            if (ty == tile_y) {
                chunk.tiles[tile].tile_neighbours[TILE_TOP] = EXTERNAL_TILE;
                chunk.tiles[tile].external_tile_mask[TILE_TOP] = 1;
            }

            if (tx <= chunk_mod_x) add_x_prev++;

            chunk.tiles[tile].t_xmin = 1;
            chunk.tiles[tile].t_xmax = right - left + 1;
            chunk.tiles[tile].t_ymin = 1;
            chunk.tiles[tile].t_ymax = top - bottom + 1;

            chunk.tiles[tile].t_left = left;
            chunk.tiles[tile].t_right = right;
            chunk.tiles[tile].t_top = top;
            chunk.tiles[tile].t_bottom = bottom;

            tile++;
        }
        add_x_prev = 0;
        if (ty <= chunk_mod_y) add_y_prev++;
    }
}

void clover_decompose(int x_cells,
                      int y_cells,
                      int* left,
                      int* right,
                      int* bottom,
                      int* top)
{
    double mesh_ratio = (double)x_cells / y_cells;
    int chunk_x = number_of_chunks;
    int chunk_y = 1;
    double factor_x, factor_y;

    int split_found = 0;
    for (int c = 1; c <= number_of_chunks; c++) {
        if (number_of_chunks % c == 0) {
            factor_x = number_of_chunks / (double)c;
            factor_y = c;

            if (factor_x / factor_y <= mesh_ratio) {
                chunk_y = c;
                chunk_x = number_of_chunks / c;
                split_found = 1;
            }
        }
    }


    if (split_found == 0 || chunk_y == number_of_chunks) {
        if (mesh_ratio >= 1.0) {
            chunk_x = number_of_chunks;
            chunk_y = 1;
        } else {
            chunk_x = 1;
            chunk_y = number_of_chunks;
        }
    }

    int delta_x = x_cells / chunk_x;
    int delta_y = y_cells / chunk_y;

    int mod_x = x_cells % chunk_x;
    int mod_y = y_cells % chunk_y;

    int add_x_prev = 0,
        add_y_prev = 0,
        add_x, add_y;

    int cnk = 1;
    for (int cy = 1; cy <= chunk_y; cy++) {
        for (int cx = 1; cx <= chunk_x; cx++) {
            add_x = 0;
            add_y = 0;

            if (cx <= mod_x) add_x = 1;
            if (cy <= mod_y) add_y = 1;

            if (cnk == parallel.task + 1) {
                *left = (cx - 1) * delta_x + 1 + add_x_prev;
                *right = *left + delta_x - 1 + add_x;
                *bottom = (cy - 1) * delta_y + 1 + add_y_prev;
                *top = *bottom + delta_y - 1 + add_y;

                chunk.chunk_neighbours[CHUNK_LEFT]   = chunk_x * (cy - 1) + cx - 1;
                chunk.chunk_neighbours[CHUNK_RIGHT]  = chunk_x * (cy - 1) + cx + 1;
                chunk.chunk_neighbours[CHUNK_BOTTOM] = chunk_x * (cy - 2) + cx;
                chunk.chunk_neighbours[CHUNK_TOP]    = chunk_x * (cy) + cx;

                if (cx == 1) chunk.chunk_neighbours[CHUNK_LEFT] = EXTERNAL_FACE;
                if (cx == chunk_x) chunk.chunk_neighbours[CHUNK_RIGHT] = EXTERNAL_FACE;
                if (cy == 1) chunk.chunk_neighbours[CHUNK_BOTTOM] = EXTERNAL_FACE;
                if (cy == chunk_y) chunk.chunk_neighbours[CHUNK_TOP] = EXTERNAL_FACE;
            }

            if (cx <= mod_x) add_x_prev++;
            cnk = cnk + 1;
        }
        add_x_prev = 0;
        if (cy <= mod_y) add_y_prev++;
    }

    BOSSPRINT(g_out, "\nMesh ratio of %.4f\n", mesh_ratio);
    BOSSPRINT(g_out, "Decomposing the mesh into %d by %d chunks\n", chunk_x, chunk_y);
    BOSSPRINT(g_out, "Decomposing the chunk with %d tiles\n\n", tiles_per_chunk);
}

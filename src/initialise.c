
#include "definitions_c.h"
#include <stdio.h>
#include "report.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "start.h"
#include "clover.h"
#include <sys/types.h>

void read_config(FILE* in);

void initialise()
{
    FILE *out_unit,
         *uin;

    g_out = fopen("./clover.out", "w");
    if (g_out == NULL) {
        report_error("initialise", "Error opening clover.out file.");
    }

    BOSSPRINT(g_out, "Clover Version %.3f", g_version);
    BOSSPRINT(stdout, "Output file clover.out opened. All output will go there.\n");
    clover_barrier();
    BOSSPRINT(g_out, "Clover will run from the following input:-\n");

    uin = fopen("./clover.in", "r");
    if (uin == NULL) {
        out_unit = fopen("clover.in", "w");
        fprintf(out_unit, "*clover");
        fprintf(out_unit, " state 1 density=0.2 energy=1.0");
        fprintf(out_unit, " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0");
        fprintf(out_unit, " x_cells=10");
        fprintf(out_unit, " y_cells=2");
        fprintf(out_unit, " xmin=0.0");
        fprintf(out_unit, " ymin=0.0");
        fprintf(out_unit, " xmax=10.0");
        fprintf(out_unit, " ymax=2.0");
        fprintf(out_unit, " initial_timestep=0.04");
        fprintf(out_unit, " timestep_rise=1.5");
        fprintf(out_unit, " max_timestep=0.04");
        fprintf(out_unit, " end_time=3.0");
        fprintf(out_unit, " test_problem 1");
        fprintf(out_unit, "*endclover");
        uin = fopen("./clover.in", "r");
    }
    clover_barrier();

    read_config(uin);

    clover_barrier();

    start();

    clover_barrier();

    BOSSPRINT(g_out, "Starting calculation\n");

    fclose(uin);
}

void parse_line(char *line, ssize_t n);
char *trimwhitespace(char *str);
int max(int a, int b);

bool strIsEqual(char *a, char *b)
{
    return strncmp(a, b, strlen(b) - 1) == 0;
}

void read_config(FILE* in)
{
    test_problem = 0;

    // int state_max = 0;

    grid.xmin =  0.0;
    grid.ymin =  0.0;
    grid.xmax = 100.0;
    grid.ymax = 100.0;

    grid.x_cells = 10;
    grid.y_cells = 10;

    end_time = 10.0;
    end_step = g_ibig;
    complete = false;

    visit_frequency = 0;
    summary_frequency = 10;

    tiles_per_chunk = 1;

    dtinit = 0.1;
    dtmax = 1.0;
    dtmin = 0.0000001;
    dtrise = 1.5;
    dtc_safe = 0.7;
    dtu_safe = 0.5;
    dtv_safe = 0.5;
    dtdiv_safe = 0.7;

    use_fortran_kernels = false;
    use_C_kernels = true;
    use_OA_kernels = false;
    profiler_on = false;
    profiler.timestep = 0.0;
    profiler.acceleration = 0.0;
    profiler.PdV = 0.0;
    profiler.cell_advection = 0.0;
    profiler.mom_advection = 0.0;
    profiler.viscosity = 0.0;
    profiler.ideal_gas = 0.0;
    profiler.visit = 0.0;
    profiler.summary = 0.0;
    profiler.reset = 0.0;
    profiler.revert = 0.0;
    profiler.flux = 0.0;
    profiler.tile_halo_exchange = 0.0;
    profiler.self_halo_exchange = 0.0;
    profiler.mpi_halo_exchange = 0.0;

    BOSSPRINT(g_out, "Reading input file\n");

    char *line = NULL;
    ssize_t n;
    size_t len = 0;
    int maxState = 0;

    while ((n = getline(&line, &len, in)) != -1) {
        char *trimmedline = trimwhitespace(line);
        if (strlen(trimmedline) >= 5) {
            if (strncmp(trimmedline, "state", 5) == 0) {
                int n = atoi(strtok(&trimmedline[5], " "));
                maxState = max(maxState, n);
            }
        }
    }
    number_of_states = maxState;

    states = malloc(sizeof(struct state_type) * maxState);
    for (int i = 0; i < maxState; i++) {
        states[i].defined = false;
        states[i].energy = 0.0;
        states[i].density = 0.0;
        states[i].xvel = 0.0;
        states[i].yvel = 0.0;
    }

    rewind(in);

    while ((n = getline(&line, &len, in)) != -1) {
        char *trimmedline = trimwhitespace(line);
        if (strlen(trimmedline) <= 1) continue;

        // printf("%s\n", trimmedline);
        if (strIsEqual(trimmedline, "initial_timestep")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            dtinit = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "initial_timestep", val);
        } else if (strIsEqual(trimmedline, "max_timestep")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            dtmax = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "max_timestep", val);
        } else if (strIsEqual(trimmedline, "timestep_rise")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            dtrise = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "timestep_rise", val);
        } else if (strIsEqual(trimmedline, "end_time")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            end_time = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "end_time", val);
        } else if (strIsEqual(trimmedline, "end_step")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            end_step = val;
            BOSSPRINT(g_out, "%25s %d\n", "end_step", val);
        } else if (strIsEqual(trimmedline, "xmin")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            grid.xmin = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "xmin", val);
        } else if (strIsEqual(trimmedline, "xmax")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            grid.xmax = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "xmax", val);
        } else if (strIsEqual(trimmedline, "ymin")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            grid.ymin = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "ymin", val);
        } else if (strIsEqual(trimmedline, "ymax")) {
            char *word = strtok(trimmedline, "=");
            double val = atof(&trimmedline[strlen(word) + 1]);
            grid.ymax = val;
            BOSSPRINT(g_out, "%25s %.4e\n", "ymax", val);
        } else if (strIsEqual(trimmedline, "x_cells")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            grid.x_cells = val;
            BOSSPRINT(g_out, "%25s %d\n", "x_cells", val);
        } else if (strIsEqual(trimmedline, "y_cells")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            grid.y_cells = val;
            BOSSPRINT(g_out, "%25s %d\n", "y_cells", val);
        } else if (strIsEqual(trimmedline, "visit_frequency")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            visit_frequency = val;
            BOSSPRINT(g_out, "%25s %d\n", "visit_frequency", val);
        } else if (strIsEqual(trimmedline, "summary_frequency")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            summary_frequency = val;
            BOSSPRINT(g_out, "%25s %d\n", "summary_frequency", val);
        } else if (strIsEqual(trimmedline, "tiles_per_chunk")) {
            char *word = strtok(trimmedline, " ");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            tiles_per_chunk = val;
            BOSSPRINT(g_out, "%25s %d\n", "tiles_per_chunk", val);
        } else if (strIsEqual(trimmedline, "tiles_per_problem")) {
            char *word = strtok(trimmedline, "=");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            tiles_per_chunk = val / 1;
            BOSSPRINT(g_out, "%25s %d\n", "tiles_per_problem", val);
        } else if (strIsEqual(trimmedline, "use_fortran_kernels")) {

        } else if (strIsEqual(trimmedline, "use_c_kernels")) {

        } else if (strIsEqual(trimmedline, "use_oa_kernels")) {

        } else if (strIsEqual(trimmedline, "profiler_on")) {
            profiler_on = true;
            BOSSPRINT(g_out, "%25s\n", "Profiler on");
        } else if (strIsEqual(trimmedline, "test_problem")) {
            char *word = strtok(trimmedline, " ");
            int val = atoi(&trimmedline[strlen(word) + 1]);
            test_problem = val;
            BOSSPRINT(g_out, "%25s %d\n", "test_problem", val);
        } else if (strIsEqual(trimmedline, "state")) {
            // char *word = strtok(trimmedline, " ");

            char *token;
            token = strsep(&trimmedline, " ");
            token = strsep(&trimmedline, " ");
            int state = atoi(token) - 1;

            BOSSPRINT(g_out, "%25s %d\n", "Reading specification for state", state + 1);

            if (states[state].defined) report_error("read_input", "State defined twice.");
            states[state].defined = true;


            while ((token = strsep(&trimmedline, " "))) {
                char *name = strsep(&token, "=");
                char *val = strsep(&token, "=");

                if (strIsEqual(name, "density")) {
                    states[state].density = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "density", atof(val));
                } else if (strIsEqual(name, "energy")) {
                    states[state].energy = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "energy", atof(val));
                } else if (strIsEqual(name, "geometry")) {
                    BOSSPRINT(g_out, "%25s ", "State geometry");
                    if (strIsEqual(val, "rectangle")) {
                        states[state].geometry = g_rect;
                        BOSSPRINT(g_out, "rectangular\n");
                    } else if (strIsEqual(val, "circle")) {
                        states[state].geometry = g_circ;
                        BOSSPRINT(g_out, "circular\n");
                    } else if (strIsEqual(val, "point")) {
                        states[state].geometry = g_point;
                        BOSSPRINT(g_out, "point\n");
                    } else {
                        BOSSPRINT(g_out, "Unknown geometry %s\n", val);
                    }
                } else if (strIsEqual(name, "xmin")) {
                    states[state].xmin = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "xmin", atof(val));
                } else if (strIsEqual(name, "xmax")) {
                    states[state].xmax = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "xmax", atof(val));
                } else if (strIsEqual(name, "ymin")) {
                    states[state].ymin = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "ymin", atof(val));
                } else if (strIsEqual(name, "ymax")) {
                    states[state].ymax = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "ymax", atof(val));
                } else if (strIsEqual(name, "radius")) {
                    states[state].radius = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "radius", atof(val));
                } else if (strIsEqual(name, "density")) {
                    states[state].density = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "density", atof(val));
                } else if (strIsEqual(name, "xvel")) {
                    states[state].xvel = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "xvel", atof(val));
                } else if (strIsEqual(name, "yvel")) {
                    states[state].yvel = atof(val);
                    BOSSPRINT(g_out, "%25s %.4e\n", "yvel", atof(val));
                } else {
                    BOSSPRINT(g_out, "Uknown param %s\n", name);
                }
            }
            BOSSPRINT(g_out, "\n");
        } else if (strIsEqual(trimmedline, "*clover")) {

        } else if (strIsEqual(trimmedline, "*endclover")) {

        } else {
            printf("Uknown command %s\n", trimmedline);
        }
    }


    // If a state boundary falls exactly on a cell boundary then round off can
    // cause the state to be put one cell further that expected. This is compiler
    // /system dependent. To avoid this, a state boundary is reduced/increased by a 100th
    // of a cell width so it lies well with in the intended cell.
    // Because a cell is either full or empty of a specified state, this small
    // modification to the state extents does not change the answers.
    float dx = (grid.xmax - grid.xmin) / (float)(grid.x_cells);
    float dy = (grid.ymax - grid.ymin) / (float)(grid.y_cells);
    for (int n = 1; n < number_of_states; n++) {
        states[n].xmin = states[n].xmin + (dx / 100.0);
        states[n].ymin = states[n].ymin + (dy / 100.0);
        states[n].xmax = states[n].xmax - (dx / 100.0);
        states[n].ymax = states[n].ymax - (dy / 100.0);
    }
}

int max(int a, int b)
{
    return a > b ? a : b;
}

// Note: This function returns a pointer to a substring of the original string.
// If the given string was allocated dynamically, the caller must not overwrite
// that pointer with the returned value, since the original pointer must be
// deallocated using the same allocator with which it was allocated.  The return
// value must NOT be deallocated using free() etc.
char *trimwhitespace(char *str)
{
    char *end;

    // Trim leading space
    while (isspace(*str)) str++;

    if (*str == 0) // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) end--;

    // Write new null terminator
    *(end + 1) = 0;

    return str;
}


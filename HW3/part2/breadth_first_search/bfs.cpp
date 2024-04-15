#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>


#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

#define alpha 48
#define beta 16


void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
	
	#pragma omp parallel proc_bind(spread)
	{

		int temp_count = 0;
		Vertex* temp_frontier = (Vertex*)malloc(sizeof(Vertex)*g->num_nodes);

		#pragma omp  for schedule(static,4)
    	for (int i = 0; i < frontier->count; i++)
    	{
	
    	    int node = frontier->vertices[i];
	
    	    int start_edge = g->outgoing_starts[node];
    	    int end_edge = (node == g->num_nodes - 1)
    	                       ? g->num_edges
    	                       : g->outgoing_starts[node + 1];
			      
    	    for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
    	    {
    	        int outgoing = g->outgoing_edges[neighbor];
				
    	        if (distances[outgoing] == NOT_VISITED_MARKER){
    	        	if(__sync_bool_compare_and_swap( &distances[outgoing] , NOT_VISITED_MARKER , distances[node] + 1 ))
    	        	{
    	        		temp_frontier[temp_count] = outgoing;
    	        		temp_count++;
    	        	}
    	        }
    	    }
    	}
    	#pragma omp critical
    	{
    		memcpy(new_frontier->vertices + new_frontier->count , temp_frontier , sizeof(int) * temp_count);
    	  	new_frontier->count+=temp_count;
    	}
    	free(temp_frontier);
	}
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for simd proc_bind(spread)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
       
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
		
        vertex_set_clear(new_frontier);
		
        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
      
       
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
      
    }
}


void bfs_bottom_up_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances) 
{
	int dist = distances[frontier->vertices[frontier->count-1]];
    #pragma omp parallel proc_bind(spread)
    {
    	int temp_count = 0;
		Vertex* temp_frontier = (Vertex*)malloc(sizeof(Vertex)*g->num_nodes);
    
    	#pragma omp  for
    	for (int v = 0; v < g->num_nodes; v++) {
    	    if (distances[v] == NOT_VISITED_MARKER) {
				/*
				int start = g->incoming_starts[v];
				int end = (v == g->num_nodes-1) 
										? g->num_edges
										: g->incoming_starts[ v+1 ];

                    

    	        for (int u = start ; u < end ; ++u) {

    	        	int incoming = g->incoming_edges[u];
				
    	        	if(distances[incoming] == dist){
    	                   		distances[v] = distances[incoming] + 1;
    	                    	temp_frontier[temp_count] = v;
    	                    	temp_count++;

    	                    	break; 
    	            }
    	            
    	        }
    	        */
    	        auto begin = incoming_begin(g, v);
    	        auto end = incoming_end(g, v);
    	        
    	        for(auto u=begin ; u!=end ; u++){
    	        	if(distances[*u] == dist){
    	        		distances[v] = dist + 1;
    	        		temp_frontier[temp_count] = v;
    	        		temp_count++;
    	        		break;
    	        	}
    	        }
    	    }
    	}
    	#pragma omp critical
    	{
    		memcpy(new_frontier->vertices + new_frontier->count , temp_frontier , sizeof(int) * temp_count);
    	  	new_frontier->count+=temp_count;
    	}
    	free(temp_frontier);
    }
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
	

    #pragma omp parallel for simd proc_bind(spread)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count !=0 ) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

    	vertex_set_clear(new_frontier);

        bfs_bottom_up_step(graph, frontier, new_frontier, sol->distances);


#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        
    }
    
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // Initialization
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    
    //switch top and bottom
	int mode = 0 , avg_out = 0;
	
	//avg outdegree
	#pragma omp parallel for simd reduction(+:avg_out)
	for(int i=0 ; i<graph->num_nodes ; i++){
		avg_out += outgoing_size(graph,i);
	}
	avg_out /= graph->num_nodes;
    
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        
        if(mode==0){
        	top_down_step(graph,frontier,new_frontier,sol->distances);
        }else{
        	bfs_bottom_up_step(graph,frontier,new_frontier,sol->distances);
        }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
		int mf=0,mu=0,nf=0,n=0;
		
		double res = (double)(new_frontier->count - frontier->count) / frontier -> count;
		
		mf = avg_out * new_frontier->count;
		mu = avg_out * (graph->num_nodes - frontier->count);
		

		nf = new_frontier -> count;
		n = graph->num_nodes;
		if(mf > (mu)/alpha){
			mode = 1;
		}else if(nf < (n/beta)){
			mode = 0;
		}

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

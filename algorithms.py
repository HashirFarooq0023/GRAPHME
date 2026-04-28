import collections
import heapq

def dfs(graph, start, goal=None):
    """
    Depth First Search (DFS) algorithm that returns state steps for visualization.
    Uses a Stack (LIFO) for traversal.
    """
    steps = []
    stack = [(start, [start])]
    visited = []
    visited_set = set()
    
    while stack:
        current, path = stack.pop()
        
        # Avoid revisiting nodes
        if current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            
            # The frontier represents all nodes currently in the stack waiting to be explored
            frontier = [n for n, p in stack]
            
            steps.append({
                "current": current,
                "frontier": list(frontier),
                "visited": list(visited),
                "path": list(path)
            })
            
            # Stop early if goal is found
            if current == goal:
                break
                
            # Add neighbors to stack in reverse order for standard alphabetical tie-breaking
            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited_set:
                    stack.append((neighbor, path + [neighbor]))
                    
    return steps


def bfs(graph, start, goal=None):
    """
    Breadth First Search (BFS) algorithm that returns state steps for visualization.
    Uses a Queue (FIFO) for traversal.
    """
    steps = []
    queue = collections.deque([(start, [start])])
    visited = []
    visited_set = set()
    frontier_set = {start}
    
    while queue:
        current, path = queue.popleft()
        if current in frontier_set:
            frontier_set.remove(current)
        
        # Avoid revisiting nodes
        if current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            
            # The frontier represents all nodes currently in the queue
            frontier = [n for n, p in queue]
            
            steps.append({
                "current": current,
                "frontier": list(frontier),
                "visited": list(visited),
                "path": list(path)
            })
            
            # Stop early if goal is found
            if current == goal:
                break
                
            # Add valid neighbors to the queue
            for neighbor in graph.get(current, []):
                if neighbor not in visited_set and neighbor not in frontier_set:
                    queue.append((neighbor, path + [neighbor]))
                    frontier_set.add(neighbor)
                    
    return steps


def dls(graph, node, goal, depth_limit, path, visited, steps):
    """
    Helper function for IDS: Depth-Limited Search (DLS).
    Records recursive traversal steps.
    """
    if node not in visited:
        visited.append(node)
        
    # In recursive DFS, the true global frontier is complex to reconstruct.
    # We calculate the local unvisited neighbors for visualization.
    local_frontier = []
    if depth_limit > 0:
        local_frontier = [n for n in graph.get(node, []) if n not in visited]
        
    steps.append({
        "current": node,
        "frontier": local_frontier,
        "visited": list(visited),
        "path": list(path),
        "depth": depth_limit
    })
    
    # Stop early if goal is found
    if node == goal:
        return True
        
    # Strictly enforce depth control
    if depth_limit > 0:
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                found = dls(graph, neighbor, goal, depth_limit - 1, path + [neighbor], visited, steps)
                if found:
                    return True
                    
    return False


def iterative_dfs(graph, start, goal, max_depth):
    """
    Iterative Deepening Depth First Search (IDS) algorithm that returns state steps for visualization.
    Repeatedly performs DLS with increasing depth limits.
    """
    steps = []
    for depth in range(max_depth + 1):
        # Reset visited nodes for each new depth iteration
        visited = []
        found = dls(graph, start, goal, depth, [start], visited, steps)
        if found:
            break
    return steps


if __name__ == "__main__":
    # Small example graph
    example_graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    
    print("=== Testing DFS ===")
    dfs_steps = dfs(example_graph, 'A', 'F')
    for i, step in enumerate(dfs_steps):
        print(f"Step {i+1}: {step}")
        
    print("\n=== Testing BFS ===")
    bfs_steps = bfs(example_graph, 'A', 'F')
    for i, step in enumerate(bfs_steps):
        print(f"Step {i+1}: {step}")
        
    print("\n=== Testing IDS (max_depth=3) ===")
    ids_steps = iterative_dfs(example_graph, 'A', 'F', 3)
def ucs(graph, start, goal, weights):
    """Uniform Cost Search"""
    steps = []
    pq = [(0, start, [start])]
    visited = []
    visited_set = set()
    frontier_set = {start}
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        if current in frontier_set:
            frontier_set.remove(current)
            
        if current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            frontier = [n for _, n, _ in pq]
            steps.append({
                "current": current,
                "frontier": list(frontier),
                "visited": list(visited),
                "path": list(path),
                "cost": cost
            })
            if current == goal:
                break
                
            for neighbor in graph.get(current, []):
                if neighbor not in visited_set:
                    edge_cost = weights.get((current, neighbor), 1)
                    new_cost = cost + edge_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
                    frontier_set.add(neighbor)
    return steps

def gbfs(graph, start, goal, heuristics):
    """Greedy Best First Search"""
    steps = []
    pq = [(heuristics.get(start, 0), start, [start])]
    visited = []
    visited_set = set()
    frontier_set = {start}
    
    while pq:
        h_cost, current, path = heapq.heappop(pq)
        if current in frontier_set:
            frontier_set.remove(current)
            
        if current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            frontier = [n for _, n, _ in pq]
            steps.append({
                "current": current,
                "frontier": list(frontier),
                "visited": list(visited),
                "path": list(path),
                "heuristic": heuristics.get(current, 0)
            })
            if current == goal:
                break
                
            for neighbor in graph.get(current, []):
                if neighbor not in visited_set:
                    h = heuristics.get(neighbor, 0)
                    heapq.heappush(pq, (h, neighbor, path + [neighbor]))
                    frontier_set.add(neighbor)
    return steps

def astar(graph, start, goal, weights, heuristics):
    """A* Search"""
    steps = []
    h_start = heuristics.get(start, 0)
    pq = [(0 + h_start, 0, start, [start])]
    visited = []
    visited_set = set()
    frontier_set = {start}
    
    while pq:
        f_cost, g_cost, current, path = heapq.heappop(pq)
        if current in frontier_set:
            frontier_set.remove(current)
            
        if current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            frontier = [n for _, _, n, _ in pq]
            steps.append({
                "current": current,
                "frontier": list(frontier),
                "visited": list(visited),
                "path": list(path),
                "cost": g_cost,
                "heuristic": heuristics.get(current, 0),
                "f_cost": f_cost
            })
            if current == goal:
                break
                
            for neighbor in graph.get(current, []):
                if neighbor not in visited_set:
                    edge_cost = weights.get((current, neighbor), 1)
                    new_g_cost = g_cost + edge_cost
                    new_h_cost = heuristics.get(neighbor, 0)
                    new_f_cost = new_g_cost + new_h_cost
                    heapq.heappush(pq, (new_f_cost, new_g_cost, neighbor, path + [neighbor]))
                    frontier_set.add(neighbor)
    return steps

def ida_star(graph, start, goal, weights, heuristics):
    """Iterative Deepening A*"""
    steps = []
    threshold = heuristics.get(start, 0)
    
    def search(node, g, bound, path, visited_list, visited_set):
        f = g + heuristics.get(node, 0)
        local_frontier = [n for n in graph.get(node, []) if n not in visited_set]
        
        steps.append({
            "current": node,
            "frontier": local_frontier,
            "visited": list(visited_list),
            "path": list(path),
            "cost": g,
            "heuristic": heuristics.get(node, 0),
            "f_cost": f,
            "f_limit": bound
        })
        
        if f > bound:
            return False, f
        if node == goal:
            return True, f
            
        min_over_bound = float('inf')
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited_set:
                visited_list.append(neighbor)
                visited_set.add(neighbor)
                
                edge_cost = weights.get((node, neighbor), 1)
                found, next_bound = search(neighbor, g + edge_cost, bound, path + [neighbor], visited_list, visited_set)
                
                if found:
                    return True, next_bound
                if next_bound < min_over_bound:
                    min_over_bound = next_bound
                    
                visited_list.pop()
                visited_set.remove(neighbor)
                
        return False, min_over_bound

    while True:
        visited_list = [start]
        visited_set = {start}
        found, next_threshold = search(start, 0, threshold, [start], visited_list, visited_set)
        
        if found:
            break
        if next_threshold == float('inf'):
            break 
            
        threshold = next_threshold

    return steps

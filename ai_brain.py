import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from dotenv import load_dotenv
import os
import math
import cv2
import logging

# Load environment variables from .env file
load_dotenv()


# MQTT settings from .env file
BROKER = os.getenv("PI4_IP", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))
SENSOR_TOPIC = os.getenv('mower/sensor_data')
GPS_TOPIC = os.getenv("GPS_TOPIC", "mower/gps")
COMMAND_TOPIC = os.getenv("COMMAND_TOPIC", "mower/command")
CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "RaspberryPiPlanner")
MOWING_AREA_TOPIC = os.getenv("MOWING_AREA_TOPIC", "mower/mowing_area")

# Read the mowing area from MQTT
mowing_area_polygon = [(0, 0), (0, 10), (10, 10), (10, 0)]  # Default square area
def on_mowing_area_message(client, userdata, msg):
    global mowing_area_polygon
    mowing_area_polygon = json.loads(msg.payload.decode())
    print(f"Received mowing area: {mowing_area_polygon}")


def point_in_polygon(x, y, polygon):
    """Check if a point (x, y) is inside the polygon."""
    num_points = len(polygon)
    j = num_points - 1
    inside = False
    for i in range(num_points):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def generate_grid_from_polygon(polygon, grid_size=1):
    """Generate a grid of points within the mowing area polygon."""
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)

    grid_points = []
    for x in np.arange(min_x, max_x, grid_size):
        for y in np.arange(min_y, max_y, grid_size):
            if point_in_polygon(x, y, polygon):
                grid_points.append((x, y))
    return grid_points

grid = generate_grid_from_polygon(mowing_area_polygon)

# A* Pathfinding Algorithm
def a_star_pathfinding(start, end, grid, obstacles):
    """A* algorithm to find the shortest path from start to end."""
    open_list = []
    closed_list = []
    open_list.append(start)

    g = {start: 0}  # Cost from start to node
    f = {start: heuristic(start, end)}  # Estimated cost from start to end

    parent = {start: None}

    def neighbors(node):
        x, y = node
        # Neighboring cells (4-way movement)
        potential_neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        # Filter out neighbors that are obstacles or outside the grid
        return [n for n in potential_neighbors if n in grid and n not in obstacles]

    while open_list:
        # Get the node with the lowest f-score
        current = min(open_list, key=lambda n: f[n])

        if current == end:
            # Path has been found; reconstruct it
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        open_list.remove(current)
        closed_list.append(current)

        for neighbor in neighbors(current):
            if neighbor in closed_list:
                continue

            tentative_g = g[current] + 1  # Distance between nodes is 1
            if neighbor not in open_list or tentative_g < g[neighbor]:
                parent[neighbor] = current
                g[neighbor] = tentative_g
                f[neighbor] = g[neighbor] + heuristic(neighbor, end)

                if neighbor not in open_list:
                    open_list.append(neighbor)

    return []  # No path found

def heuristic(node1, node2):
    """Heuristic for A* (Euclidean distance)."""
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

def navigate_to_waypoints(waypoints, grid, obstacles):
    """Navigate mower through a list of waypoints using A* for obstacle avoidance."""
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        path = a_star_pathfinding(start, end, grid, obstacles)
        for step in path:
            """ Publish each movement command based on path (target location must
            be a tuple with latitude and longitude coordinates)
            """
            command = {"target_location": {"x": step[0], "y": step[1]}}
            client.publish(COMMAND_TOPIC, json.dumps(command))
            time.sleep(0.5)


def publish_path_to_mqtt(path):
    """Publish the planned path to the MQTT topic for the WebUI."""
    path_data = {"path": path}
    client.publish("mower/path", json.dumps(path_data))

def path_planning_algorithm(sensor_data, pattern_type):
    """Main path planning algorithm integrating pattern selection and A*."""
    global obstacles
    obstacles = [(int(sensor_data["position"]["x"]), int(sensor_data["position"]["y"]))]

    # Generate mowing pattern waypoints
    waypoints = create_pattern(pattern_type, grid.shape)
    # Navigate the waypoints using the A* algorithm
    full_path = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        path = a_star_pathfinding(start, end, grid, obstacles)
        full_path.extend(path)

    # Publish the complete path to the MQTT topic for the WebUI
    publish_path_to_mqtt(full_path)

    # Continue with navigating the waypoints
    navigate_to_waypoints(waypoints, grid, obstacles)


def create_pattern(pattern_type, grid_shape):
    """Create waypoints based on selected pattern."""
    waypoints = []
    width, height = grid_shape

    if pattern_type == "stripes":
        for x in range(0, width, 2):
            waypoints.append((x, 0))
            waypoints.append((x, height - 1))

    elif pattern_type == "criss_cross":
        for x in range(0, width, 2):
            waypoints.append((x, 0))
            waypoints.append((x, height - 1))
        for y in range(0, height, 2):
            waypoints.append((0, y))
            waypoints.append((width - 1, y))

    elif pattern_type == "checkerboard":
        for x in range(0, width, 4):
            for y in range(0, height, 4):
                # Create alternating blocks for the checkerboard pattern
                waypoints.append((x, y))
                waypoints.append((x + 2, y + 2))
                waypoints.append((x, y + 4))
                waypoints.append((x + 2, y + 6))

    elif pattern_type == "diamond":
        center_x = width // 2
        center_y = height // 2
        max_distance = min(center_x, center_y)
        for d in range(0, max_distance, 2):
            waypoints.append((center_x - d, center_y))
            waypoints.append((center_x, center_y - d))
            waypoints.append((center_x + d, center_y))
            waypoints.append((center_x, center_y + d))

    elif pattern_type == "waves":
        for y in range(0, height, 4):
            for x in range(width):
                offset = (x % 10) // 5  # Change offset frequency for wave effect
                if offset == 0:
                    waypoints.append((x, y))
                else:
                    waypoints.append((x, y + offset))

    elif pattern_type == "concentric_circles":
        center_x = width // 2
        center_y = height // 2
        max_radius = min(center_x, center_y)
        for r in range(2, max_radius, 3):
            waypoints.extend(circle_waypoints(center_x, center_y, r))

    elif pattern_type == "stars":
        # Define star points relative to center
        center_x, center_y = width // 2, height // 2
        for i in range(5):
            angle = np.radians(i * 144)  # Star points are spaced 144 degrees apart
            outer_x = int(center_x + (width // 4) * np.cos(angle))
            outer_y = int(center_y + (height // 4) * np.sin(angle))
            waypoints.append((outer_x, outer_y))

    elif pattern_type == "custom_image":
        img_path = os.getenv("USER_IMAGE_PATH", "image.png")
        x_offset = int(os.getenv("IMAGE_X_OFFSET", 0))
        y_offset = int(os.getenv("IMAGE_Y_OFFSET", 0))
        waypoints = image_to_waypoints(img_path, x_offset, y_offset, grid_shape)

    else:
        raise ValueError(f"Unsupported pattern type: {pattern_type}")

    return waypoints


def image_to_waypoints(img_path, x_offset, y_offset, grid_shape):
    """Convert an image to a set of waypoints."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (grid_shape[1], grid_shape[0]))

    # Threshold the image to find contours
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    waypoints = []
    for contour in contours:
        for point in contour:
            x = point[0][0] + x_offset
            y = point[0][1] + y_offset
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                waypoints.append((x, y))

    return waypoints


def circle_waypoints(center_x, center_y, radius, step=15):
    """Generate waypoints for a circle with a given radius."""
    waypoints = []
    for angle in range(0, 360, step):
        x = int(center_x + radius * np.cos(np.radians(angle)))
        y = int(center_y + radius * np.sin(np.radians(angle)))
        waypoints.append((x, y))
    return waypoints

# MQTT callbacks
def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected with result code {rc}")
    client.subscribe(SENSOR_TOPIC)
    client.subscribe(MOWING_AREA_TOPIC)
    client.subscribe(GPS_TOPIC)
    client.subscribe('mower/pattern_type')

pattern_type = "stripes"  # Default pattern type

def on_message(client, userdata, msg):
    global pattern_type
    if msg.topic == SENSOR_TOPIC:
        sensor_data = json.loads(msg.payload.decode())
        print(f"Received sensor data: {sensor_data}")
        # Use the updated pattern_type
        path_planning_algorithm(sensor_data, pattern_type)
    elif msg.topic == MOWING_AREA_TOPIC:
        on_mowing_area_message(client, userdata, msg)
    elif msg.topic == GPS_TOPIC:
        gps_data = json.loads(msg.payload.decode())
        print(f"Received GPS data: {gps_data}")
        # Process GPS data as needed
    elif msg.topic == 'mower/pattern_type':
        pattern_type = msg.payload.decode()
        print(f"Received pattern type: {pattern_type}")
    else:
        print(f"Unknown topic: {msg.topic}")

def on_publish(client, userdata, mid):
    logging.info(f"Published message ID: {mid}")

# MQTT setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_publish
client.connect(BROKER, PORT, 60)

# Main loop
client.loop_forever()

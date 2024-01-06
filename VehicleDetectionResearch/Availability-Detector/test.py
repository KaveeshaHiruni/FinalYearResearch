# Function to draw a traffic light with a black rectangle
def draw_traffic_light_with_rectangle(traffic, state):
    light_radius = 30
    light_spacing = 20
    light_x = 50
    light_y = 50
    rectangle_width = 120
    rectangle_height = 200

    # Draw the black rectangle
    cv2.rectangle(traffic, (light_x - rectangle_width // 2, light_y - rectangle_height // 2),
                  (light_x + rectangle_width // 2, light_y + rectangle_height // 2), (0, 0, 0), cv2.FILLED)

    for i, color in enumerate(traffic_light_colors):
        if i == state:
            cv2.circle(traffic, (light_x, light_y + i * (light_spacing + light_radius)), light_radius,
                       color, cv2.FILLED)
        else:
            cv2.circle(traffic, (light_x, light_y + i * (light_spacing + light_radius)), light_radius, color, 3)
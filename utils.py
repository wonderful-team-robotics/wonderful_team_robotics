import json
import math
import os
import random
import statistics
import time
import copy
import io
import re
import base64
import inspect
from io import BytesIO
import argparse
import shutil
import gc
import traceback

import yaml
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

CLIENT = OpenAI()

def draw_func_traj(rgb_array, trajectory, color='r', linewidth=4):
    height, width, _ = rgb_array.shape

    # Convert the trajectory to a numpy array for uniform processing
    trajectory = np.array(trajectory)

    # Ensure trajectory has at least 2 dimensions (for x and y)
    if trajectory.shape[1] < 2:
        return None
    
    # Extract x and y coordinates, ignoring z if present
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(rgb_array)

    # Plot the trajectory on the image
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)

    pil_image = draw_ticks(fig, ax, width, height)

    plt.close(fig)

    return pil_image

def update_system_memory(response, system_memory):
    if response["descriptive_key"] != "none":
        descriptive_key = response["descriptive_key"]
        information_summary = response["information_summary"]

        system_memory[descriptive_key] = information_summary
    
    return system_memory

def collect_logs(text_file, image_folder, video_file, env, run_number):
    shutil.move(text_file, os.path.join(image_folder, os.path.basename(text_file)))
    
    # Move the video file into the image folder
    shutil.move(video_file, os.path.join(image_folder, os.path.basename(video_file)))
    
    # Rename the image folder to the run number
    new_folder_name = os.path.join(os.path.dirname(image_folder), f"{env}_{run_number}")
    if os.path.exists(new_folder_name):
        shutil.rmtree(new_folder_name)
    os.rename(image_folder, new_folder_name)

def get_input():
    parser = argparse.ArgumentParser(description="Process some input string.")
    parser.add_argument('--task', type=str, required=True, help="The task name.")
    parser.add_argument('--env_type', type=str, required=True, help="The environment type.")
    parser.add_argument('--run_number', type=str, required=False, default="0", help="The run number.")
    parser.add_argument('--vlm', type=str, required=True, help="Visual language model name.")
    parser.add_argument('--collect_log', action='store_true', help="Whether to collect log.")
    parser.add_argument('--prompt', type=str, required=True, help="Task description.")
    # Parse the arguments
    args = parser.parse_args()
    
    return args

def get_box_margin(rgb_array, box, margin=60, direction="inside", labels=['A', 'B', 'C', 'D'], canvas_bg_color=(255, 255, 255), label_font_size=30, font_path='Arial.ttf'):
    height, width, _ = rgb_array.shape
    x, y, w, h = box
    
    left = max(x - w // 2, 0)
    top = max(y - h // 2, 0)
    right = min(left + w, width)
    bottom = min(top + h, height)

    box_w = right - left
    box_h = bottom - top
    margin = min(margin, min(box_h//2, box_w//2))
    
    # Extract the slits with the specified margin
    if direction == "inside":
        top_slit = rgb_array[bottom-margin:bottom, left:right, :]
        bottom_slit = rgb_array[top:top+margin, left:right, :]
        left_slit = rgb_array[top:bottom, left:left+margin, :]
        right_slit = rgb_array[top:bottom, right-margin:right, :]
    
    elif direction == "outside":
        black_slit_horizontal = np.zeros((margin, w, 3), dtype=np.uint8)
        black_slit_vertical = np.zeros((h, margin, 3), dtype=np.uint8)
        if bottom+margin > height:
            top_slit = black_slit_horizontal
        else:
            top_slit = rgb_array[bottom:bottom+margin, left:right, :]
        if top-margin < 0:
            bottom_slit = black_slit_horizontal
        else:
            bottom_slit = rgb_array[top-margin:top, left:right, :]
        if left-margin < 0:
            left_slit = black_slit_vertical
        else:
            left_slit = rgb_array[top:bottom, left-margin:left, :]
        if right+margin > width:
            right_slit = black_slit_vertical
        else:
            right_slit = rgb_array[top:bottom, right:right+margin, :]
    
    left_slit = np.rot90(left_slit, k=1)
    right_slit = np.rot90(right_slit, k=1)

    images = [top_slit, bottom_slit, left_slit, right_slit]
    
    # Determine the width and height for the canvas based on images and label space
    margin_between_image = 60
    max_image_width = max(image.shape[1] for image in images)
    total_height = sum(image.shape[0] for image in images) + (len(images) - 1) * margin_between_image  # Adding margin between images
    label_width = label_font_size * 2  # Estimate width of label area

    canvas_width = max_image_width + label_width + 80  # Add some padding
    canvas_height = total_height + 80  # Add some padding

    # Create an empty white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), canvas_bg_color)
    draw = ImageDraw.Draw(canvas)

    # Load a default font
    font = ImageFont.truetype(font_path, label_font_size)

    y_offset = 20  # Initial top margin

    for i, (image_array, label) in enumerate(zip(images, labels)):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        # Position to paste the image
        image_position = (label_width + 20, y_offset)
        
        # Paste the image on the canvas
        canvas.paste(image, image_position)
        
        # Position to draw the label
        label_position = (20, y_offset + (image_array.shape[0] - label_font_size) // 2)
        
        # Draw the label
        draw.text(label_position, label, fill='black', font=font)
        
        # Update y_offset for the next image
        y_offset += image_array.shape[0] + margin_between_image  # Adding margin between images

    return canvas

def update_edge_status(response, current_edge_status):
    if current_edge_status["left_edge"] == "further_adjust" and response["left_edge"] == "fixed":
        current_edge_status["left_edge"] = "fixed"

    if current_edge_status["right_edge"] == "further_adjust" and response["right_edge"] == "fixed":
        current_edge_status["right_edge"] = "fixed"

    if current_edge_status["top_edge"] == "further_adjust" and response["top_edge"] == "fixed":
        current_edge_status["top_edge"] = "fixed"

    if current_edge_status["bottom_edge"] == "further_adjust" and response["bottom_edge"] == "fixed":
        current_edge_status["bottom_edge"] = "fixed"

    return current_edge_status

def check_edge_status_to_end(current_edge_status):
    if current_edge_status["left_edge"] == "fixed" \
    and current_edge_status["right_edge"] == "fixed" \
    and current_edge_status["top_edge"] == "fixed" \
    and current_edge_status["bottom_edge"] == "fixed":
        return True
    else:
        return False

def resize_box_size(box, response):
    x, y, w, h = box
    if response["width_change_amount"] == "tiny":
        amount = 0.1
    elif response["width_change_amount"] == "small":
        amount = 0.3
    elif response["width_change_amount"] == "medium":
        amount = 0.5
    elif response["width_change_amount"] == "large":
        amount = 0.7

    if response["height_change_amount"] == "tiny":
        amount = 0.1
    elif response["height_change_amount"] == "small":
        amount = 0.3
    elif response["height_change_amount"] == "medium":
        amount = 0.5
    elif response["height_change_amount"] == "large":
        amount = 0.7

    if response["change_in_width"] == "wider":
        h *= (1 + amount)
    elif response["change_in_width"] == "narrower":
        h *= (1 - amount)
    
    if response["change_in_height"] == "taller":
        w *= (1 + amount)
    elif response["change_in_height"] == "shorter":
        w *= (1 - amount)
    
    return (x, y, w, h)

def adjust_box_position(box, response):
    x, y, w, h = box
    if response["horizontal_change_amount"] == "tiny":
        h_amount = 30
    elif response["horizontal_change_amount"] == "small":
        h_amount = 60
    elif response["horizontal_change_amount"] == "medium":
        h_amount = 100
    elif response["horizontal_change_amount"] == "large":
        h_amount = 150
    else:
        h_amount = 0

    if response["vertical_change_amount"] == "tiny":
        v_amount = 30
    elif response["vertical_change_amount"] == "small":
        v_amount = 60
    elif response["vertical_change_amount"] == "medium":
        v_amount = 100
    elif response["vertical_change_amount"] == "large":
        v_amount = 150
    else:
        v_amount = 0

    if response["change_in_horizontal_position"] == "left":
        x -= h_amount
    elif response["change_in_horizontal_position"] == "right":
        x += h_amount

    if response["change_in_vertical_position"] == "up":
        y += v_amount
    elif response["change_in_vertical_position"] == "down":
        y -= v_amount
    
    return [x, y, w, h]


def save_pil(pil_image, dir='log', file_name='', show=False):
    # Ensure the folder exists
    os.makedirs(dir, exist_ok=True)

    # Generate the filename based on the current time
    current_time = time.time()
    if len(file_name):
        file_name = file_name + '_'
    filename = f'{file_name}{current_time}.png'

    # Construct the full path
    file_path = os.path.join(dir, filename)

    # Save the image
    pil_image.save(file_path)

    if show:
        plt.imshow(pil_image)
        plt.axis('off')  
        plt.pause(2)

    return file_path

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def concat_pil_images(img_list, caption_list, text_height=60, font_path="Arial.ttf", font_size=40):
    # Ensure all images are the same width and height
    width, height = img_list[0].size
    
    # Create a new image with added height for the text
    new_height = height + text_height
    combined_img = Image.new('RGB', (width, new_height * len(img_list)), 'white')

    # Load the font
    font = ImageFont.truetype(font_path, font_size)
    
    height_offset = 0
    for i, img in enumerate(img_list):
        # Create an image with white space on top for text
        new_img = Image.new('RGB', (width, new_height), 'white')
        new_img.paste(img, (0, text_height))
        
        # Draw the text on the white space
        draw = ImageDraw.Draw(new_img)
        draw.text((10, 10), caption_list[i], fill="black", font=font)
        
        # Paste the new image onto the combined image
        combined_img.paste(new_img, (0, height_offset))
        height_offset += new_height

    return combined_img

        

def resize_rgb_array(img, width=2000):
    # Calculate the new height while maintaining the aspect ratio
    height = int(img.shape[0] * (width / img.shape[1]))
    
    # Resize the image using the specified interpolation method
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    return resized_img

def draw_ticks(fig, ax, width, height, rgb_array=None):
    if rgb_array is not None:
        height, width, _ = rgb_array.shape
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(rgb_array)

    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # Set x and y ticks
    x_ticks = np.linspace(0, width, num=11)
    y_ticks = np.linspace(round(height/100)*100, 0, num=11)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Manually set the y-tick labels to start from 0 at the bottom
    ax.set_yticklabels([int(tick) for tick in y_ticks])
    # Invert the y-axis so that it starts from 0 at the bottom
    plt.gca().invert_yaxis()
    # Show the plot
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    pil_image = Image.open(buf)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return pil_image

def pil_image_to_base64(image, format='PNG'):
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return image_base64

def rgb_to_base64(rgb_array):
    image = Image.fromarray(rgb_array.astype('uint8'), 'RGB')
    image_base64 = pil_image_to_base64(image)
    return image_base64

def adjust_edge(box, response, rgb_array):
    rgb_height, rgb_width, _ = rgb_array.shape
    x, y, w, h = box

    edges = response["adjust_edge"]
    amounts = response["adjust_amount"]

    left = x - w // 2
    top = y - h // 2
    right = left + w
    bottom = top + h

    for edge, amount in zip(edges, amounts):
        if amount == "small":
            amount = 50
        elif amount == "large":
            amount = 100
        else:
            amount = 0

        if edge == "left":
            left = left - amount
            left = max(left, 0)
        elif edge == "right":
            right = right + amount
            right = min(right, rgb_width)
        elif edge == "top":
            bottom = bottom + amount
            bottom = min(bottom, rgb_height)
        elif edge == "bottom":
            top = top - amount
            top = max(top, 0)

    new_x = (left + right) // 2
    new_y = (top + bottom) // 2
    new_w = right - left
    new_h = bottom - top

    return (new_x, new_y, new_w, new_h)


def draw_box(rgb_array, box, edgecolor='red', linewidth=2, facecolor='none', alpha=1):
    height, width, _ = rgb_array.shape
    x, y, w, h = box

    # Calculate box coordinates
    left = x - w // 2
    top = y - h // 2

    # Create a plot
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(rgb_array)

    # Draw the rectangle
    rect = plt.Rectangle((left, top), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
    ax.add_patch(rect)

    pil_image = draw_ticks(fig, ax, width, height)

    return pil_image

def draw_box_zoomed(rgb_array, box, edgecolor='red', linewidth=2, padding=200, tick_size=10, facecolor='none', alpha=1, prev_box=None):
    height, width, _ = rgb_array.shape
    x, y, w, h = box

    # Calculate box coordinates
    left = max(0, x - w // 2)
    top = max(0, y - h // 2)
    right = min(width, x + w // 2)
    bottom = min(height, y + h // 2)

    if prev_box is None:
        left_zoomed = max(0, left - padding)
        top_zoomed = max(0, top - padding)
        right_zoomed = min(width, right + padding)
        bottom_zoomed = min(height, bottom + padding)
        cropped_array = rgb_array[int(top_zoomed):int(bottom_zoomed), int(left_zoomed):int(right_zoomed)]
        rect_x = left - left_zoomed
        rect_y = top - top_zoomed
    else:
        prev_x, prev_y, prev_w, prev_h = prev_box
        prev_left = max(0, prev_x - prev_w // 2)
        prev_top = max(0, prev_y - prev_h // 2)
        prev_right = min(width, prev_x + prev_w // 2)
        prev_bottom = min(height, prev_y + prev_h // 2)
        prev_left_zoomed = max(0, prev_left - padding)
        prev_top_zoomed = max(0, prev_top - padding)
        prev_right_zoomed = min(width, prev_right + padding)
        prev_bottom_zoomed = min(height, prev_bottom + padding)
        cropped_array = rgb_array[int(prev_top_zoomed):int(prev_bottom_zoomed), int(prev_left_zoomed):int(prev_right_zoomed)]
        rect_x = left - prev_left_zoomed
        rect_y = top - prev_top_zoomed

    # Create a plot
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(cropped_array)

    # Set ticks corresponding to the original scale
    def calculate_ticks(start, end):
        ticks = [i for i in range(((start // 100) + 1) * 100, (end // 100 + 1) * 100, 100)]
        return ticks

    if prev_box is None:
        x_ticks_shifted = calculate_ticks(left_zoomed, right_zoomed)
        y_ticks_shifted = calculate_ticks(top_zoomed, bottom_zoomed)
        
        x_ticks = [x - left_zoomed for x in x_ticks_shifted]
        y_ticks = [y - top_zoomed for y in y_ticks_shifted]
    else:
        x_ticks_shifted = calculate_ticks(prev_left_zoomed, prev_right_zoomed)
        y_ticks_shifted = calculate_ticks(prev_top_zoomed, prev_bottom_zoomed)
        
        x_ticks = [x - prev_left_zoomed for x in x_ticks_shifted]
        y_ticks = [y - prev_top_zoomed for y in y_ticks_shifted]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_shifted, fontsize=tick_size)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_shifted, fontsize=tick_size)
    plt.gca().invert_yaxis()

    # Draw the rectangle
    rect = plt.Rectangle((rect_x, rect_y), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
    ax.add_patch(rect)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Load the BytesIO object into a PIL Image
    pil_image = Image.open(buf)

    plt.close(fig)

    return pil_image

def draw_point_zoomed(rgb_array, box, linewidth=2, tick_size=10):
    height, width, _ = rgb_array.shape
    x, y, w, h = box

    # Calculate box coordinates
    left = max(0, x - w // 2)
    top = max(0, y - h // 2)
    right = min(width, x + w // 2)
    bottom = min(height, y + h // 2)

    w_margin = int(w * 0.2)
    h_margin = int(h * 0.2)

    left_zoomed = max(0, left - w_margin)
    top_zoomed = max(0, top - h_margin)
    right_zoomed = min(width, right + w_margin)
    bottom_zoomed = min(height, bottom + h_margin)
    cropped_array = rgb_array[int(top_zoomed):int(bottom_zoomed), int(left_zoomed):int(right_zoomed)]

    # Create a plot
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(cropped_array)

    # Set ticks corresponding to the original scale
    def calculate_ticks(start, end):
        ticks = [i for i in range(((start // 100) + 1) * 100, (end // 100 + 1) * 100, 100)]
        return ticks

    x_ticks_shifted = calculate_ticks(left_zoomed, right_zoomed)
    y_ticks_shifted = calculate_ticks(top_zoomed, bottom_zoomed)
    
    x_ticks = [x - left_zoomed for x in x_ticks_shifted]
    y_ticks = [y - top_zoomed for y in y_ticks_shifted]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_shifted, fontsize=tick_size)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_shifted, fontsize=tick_size)
    plt.gca().invert_yaxis()

    # Calculate the center and halfway points
    center_x = (right - left) / 2
    center_y = (bottom - top) / 2
    halfway_points = [
        (center_x, center_y),  # Center
        (center_x / 2, center_y / 2),  # Top-left halfway
        (3 * center_x / 2, center_y / 2),  # Top-right halfway
        (center_x / 2, 3 * center_y / 2),  # Bottom-left halfway
        (3 * center_x / 2, 3 * center_y / 2),  # Bottom-right halfway
        (center_x, center_y / 5), # center top
        (center_x, 9 * center_y / 5), # center bottom
        (center_x / 5, center_y), # center left
        (9 * center_x / 5, center_y), # center right
    ]

    # Draw semi-transparent circles and number labels
    points_dict = {}
    for i, (hx, hy) in enumerate(halfway_points):
        points_dict[i + 1] = (hx + left, hy + top)
        circle = Circle((hx+w_margin, hy+h_margin), radius=int(0.1*min(w,h)), color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.text(hx+w_margin, hy+h_margin, str(i+1), color='white', fontsize=24, ha='center', va='center')

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Load the BytesIO object into a PIL Image
    pil_image = Image.open(buf)

    plt.close(fig)

    return pil_image, points_dict

def draw_box_with_inset(rgb_array, box, edgecolor='red', linewidth=2, facecolor='none', alpha=1):
    height, width, _ = rgb_array.shape
    x, y, w, h = box

    # Calculate box coordinates
    left = max(0, x - w // 2)
    top = max(0, y - h // 2)
    right = min(width, x + w // 2)
    bottom = min(height, y + h // 2)

    # Crop the image
    cropped_array = rgb_array[int(top):int(bottom), int(left):int(right)]

    # Create a plot
    fig, ax = plt.subplots()

    # Display the full image
    ax.imshow(rgb_array)

    # Draw the rectangle on the full image
    rect = plt.Rectangle((left, top), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
    ax.add_patch(rect)

    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # Set x and y ticks
    x_ticks = np.linspace(0, width, num=11)
    y_ticks = np.linspace(height, 0, num=11)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Manually set the y-tick labels to start from 0 at the bottom
    ax.set_yticklabels([int(tick) for tick in y_ticks])
    # Invert the y-axis so that it starts from 0 at the bottom
    plt.gca().invert_yaxis()

    # Create an inset axis
    inset_size = 0.4
    inset_x = 0.95
    inset_y = 0.3
    ax_inset = fig.add_axes([inset_x, inset_y, inset_size, inset_size])
    ax_inset.imshow(cropped_array)

    # Set ticks corresponding to the original scale
    ax_inset.set_xticks([0, cropped_array.shape[1]])
    ax_inset.set_xticklabels([left, right])
    ax_inset.set_yticks([0, cropped_array.shape[0]])
    ax_inset.set_yticklabels([top, bottom])

    # invert y axis
    plt.gca().invert_yaxis()


    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Load the BytesIO object into a PIL Image
    pil_image = Image.open(buf)

    plt.close(fig)

    return pil_image


def draw_center(rgb_array, center):
    height, width, _ = rgb_array.shape

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(rgb_array)

    if isinstance(center[0], list):
        for i, point in enumerate(center):
            cx, cy = point
            if i == 0:
                circle = Circle((cx, cy), radius=40, color='red', alpha=0.8)
            if i == 1:
                circle = Circle((cx, cy), radius=40, color='green', alpha=0.8)
            if i == 2:
                circle = Circle((cx, cy), radius=40, color='blue', alpha=0.8)
            ax.add_patch(circle)
            ax.text(cx, cy, str(i), color='white', fontsize=15, ha='center', va='center')
    else:
        cx, cy = center

        ax.plot(cx, cy, marker='o', markersize=15, markeredgewidth=1, markeredgecolor='k', markerfacecolor='w')
        ax.plot(cx, cy, marker='*', markersize=12, markeredgewidth=1, markeredgecolor='k', markerfacecolor='r')

    pil_image = draw_ticks(fig, ax, width, height)

    return pil_image

def adjust_margin(box, response, margin=60):
    x, y, w, h = box

    inside_margin = response["inside_margin"]
    for key, value in inside_margin.items():
        if isinstance(value, str):
            inside_margin[key] = (value == "True")
    outside_margin = response["outside_margin"]
    for key, value in outside_margin.items():
        if isinstance(value, str):
            outside_margin[key] = (value == "True")

    for edge in list(inside_margin.keys()):
        if edge == "A":
            if outside_margin[edge]:
                y += margin // 2
                h += margin
            elif not inside_margin[edge]:
                y -= margin // 2
                h -= margin
        elif edge == "B":
            if outside_margin[edge]:
                y -= margin // 2
                h += margin
            elif not inside_margin[edge]:
                y += margin // 2
                h -= margin
        elif edge == "C":
            if outside_margin[edge]:
                x -= margin // 2
                w += margin
            elif not inside_margin[edge]:
                x += margin // 2
                w -= margin 
        elif edge == "D":
            if outside_margin[edge]:
                x += margin // 2
                w += margin
            elif not inside_margin[edge]:
                x -= margin // 2
                w -= margin
            
    if w < 30:
        w = 100
    if h < 30:
        h = 100
    if (x, y, w, h) == box:
        done = True
    else:
        done = False
        
    return (x, y, w, h), done

def degrees_to_quaternion(degrees):
    # Convert degrees to radians
    radians = math.radians(degrees)
    
    # Calculate the quaternion components
    w = math.cos(radians / 2)
    x = 0
    y = 0
    z = math.sin(radians / 2)
    
    # Normalize the quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    
    return (x, y, z, w)

def create_messages(prompt, persona, conversation_history="", image_list=[]):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Assistant Persona\n\n" + persona},
            ],
        },
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Conversation History\n\n" + conversation_history},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if len(image_list):
        for base64_image in image_list:
            messages[2]['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
                })
    return messages

def format_dict(d, indent=4):
    formatted_str = ''
    for key, value in d.items():
        formatted_str += ' ' * indent + str(key) + ': '
        if isinstance(value, dict):
            formatted_str += '\n' + format_dict(value, indent + 4)
        else:
            formatted_str += str(value) + '\n'
    return formatted_str

def prepare_dir(folder_name, file_name):
    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")
    
    with open(file_name, 'w') as file:
        # Write some content to the file
        file.write("GPT Response Log File\n\n\n")

def write_to_log(file_name, task, text):
    if isinstance(text, dict):
        text = format_dict(text)
    else:
        text = str(text)
    with open(file_name, 'a') as file:
        file.write('\n\n\n\n' + task)
        file.write('\n\n' + text)

class ConversationMemory:
    def __init__(self):
        self.memory = []

    def add_to_memory(self, sender: str, responder: str, message: str, response: str):
        if len(self.memory) >= 10:
            self.memory.pop(0)

        self.memory.append({
            "sender": sender,
            "responder": responder,
            "message": message,
            "response": response
        })

    def get_memory(self) -> str:
        memory_context = "\n".join(
            [f"{entry['sender']}:\n{entry['message']}\n{entry['responder']}:\n{entry['response']}" 
             for entry in self.memory]
        )

        return memory_context

class LMAgent:
    main_task = ""
    def __init__(self, supervisor, role, persona, vlm="gpt-4o"):
        self.vlm = vlm
        self.conv_memory = ConversationMemory()
        self.supervisor = supervisor
        self.role = role
        self.persona = persona
        self.tool_list = []
        self.inbox = {'prompt': "", "image_list": None}
    
    def set_main_task(self, task):
        self.main_task = task

    def send_communication(self, agent, prompt, image_list=[]):
        agent.receive_communication(prompt, image_list)

    def receive_communication(self, prompt, image_list=[]):
        self.inbox['prompt'] = prompt
        self.inbox['image_list'] = image_list

    def process_task(self, remember=True):
        response = self.get_response(self.inbox['prompt'], self.inbox['image_list'], remember=remember)
        return response

    def get_response(self, prompt, image_list=[], remember=True):
        conversation_history = self.conv_memory.get_memory()
        if "gpt" in self.vlm:
            messages = create_messages(prompt, self.persona, conversation_history, image_list)
        max_retries = 6
        retries = 0
        while retries < max_retries:
            retries += 1
            try:
                if "gpt" in self.vlm:
                    response = CLIENT.chat.completions.create(
                        model=self.vlm,
                        messages=messages,
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                    )
                    response = json.loads(response.choices[0].message.content)
                break
            except Exception as e:
                print(f"An error had occurred: {e}.\nNumber of attempts: {retries}")

        if remember:
            self.conv_memory.add_to_memory(sender=self.supervisor, 
                                        responder=self.role, 
                                        message=prompt,
                                        response=response)
        return response

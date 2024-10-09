from utils import *

def execute_code(code_str):
    var_before = locals().copy()
    try:
        exec(code_str, globals(), locals())
    except Exception as e:
        return f"An error occurred: {e}"
    var_after = locals().copy()
    added = {k: var_after[k] for k in var_after if k not in var_before or k != "var_before"}

    return added


def main(env, image_dir, text_log, vlm, args):
    #***********************************************************#

    text_prompt = "Wipe the plate with the sponge in simple circular motion."

    top_camera = TopCamera()
    thread = threading.Thread(target=top_camera.spin)
    thread.start()

    rospy.sleep(1)

    image, depth_array = top_camera.render()

    depth_array = np.flipud(np.array(depth_array))
    image_number = 0
    pil_image = draw_ticks('', '', '', '', depth_array)
    save_pil(pil_image, dir=image_dir, file_name="Environment_Image_DEPTH")

    rgb_array = np.flipud(np.array(image))
    image_number = 0
    pil_image = draw_ticks('', '', '', '', rgb_array)
    save_pil(pil_image, dir=image_dir, file_name="Environment_Image_RGB")

    base64_top = pil_image_to_base64(pil_image)
    image_prompt = [base64_top]
    #***********************************************************#

    write_to_log(text_log, "ENV", env)

    llm_prompt = load_config("llm_prompt.yaml")

    supervisor_persona = llm_prompt["supervisor"]["assistant_persona"]

    memory_agent_persona = llm_prompt["memory_agent"]["assistant_persona"]

    verification_agent_persona = llm_prompt["verification_agent"]["assistant_persona"]
    
    grounding_manager_persona = llm_prompt["grounding_manager"]["assistant_persona"]

    box_checker_persona = llm_prompt["box_checker"]["assistant_persona"]
    box_mover_persona = llm_prompt["box_mover"]["assistant_persona"]

    supervisor_create_plan = llm_prompt["supervisor"]["create_plan"]
    supervisor_revise_plan = llm_prompt["supervisor"]["revise_plan"]
    supervisor_request_function = llm_prompt["supervisor"]["request_function"]
    supervisor_generate_function = llm_prompt["supervisor"]["generate_function"]
    supervisor_run_function = llm_prompt["supervisor"]["run_function"]
    supervisor_verify_execution = llm_prompt["supervisor"]["verify_execution"]
    supervisor_convert_actions_to_sequence = llm_prompt["supervisor"]["convert_actions_to_sequence"]

    memory_agent_update_memory = llm_prompt["memory_agent"]["update_memory"]

    verification_agent_check_subgoal = llm_prompt["verification_agent"]["check_subgoal"]
    verification_agent_extract_targets = llm_prompt["verification_agent"]["extract_targets"]
    verification_agent_check_action_sequence = llm_prompt["verification_agent"]["check_action_sequence"]

    grounding_manager_identify_initial_center = llm_prompt["grounding_manager"]["identify_initial_center"]
    grounding_manager_select_initial_center = llm_prompt["grounding_manager"]["select_initial_center"]
    grounding_manager_identify_initial_bbox = llm_prompt["grounding_manager"]["identify_initial_bbox"]
    grounding_manager_identify_area_point = llm_prompt["grounding_manager"]["identify_area_point"]
    grounding_manager_select_best_area_point = llm_prompt["grounding_manager"]["select_best_area_point"]
    grounding_manager_select_bset_object_point = llm_prompt["grounding_manager"]["select_bset_object_point"]

    box_checker_check_revision = llm_prompt["box_checker"]["check_revision"]

    box_mover_adjust_margin = llm_prompt["box_mover"]["adjust_margin"]
    box_mover_move_box_positon = llm_prompt["box_mover"]["move_box_position"]

    replan = llm_prompt["replan"]


    supervisor = LMAgent(supervisor="human_user", role="supervisor", persona=supervisor_persona, vlm=vlm)
    memory_agent = LMAgent(supervisor="human_user", role="memory_agent", persona=memory_agent_persona, vlm=vlm)
    verification_agent = LMAgent(supervisor="supervisor", role="verification_agent", persona=verification_agent_persona, vlm=vlm)
    grounding_manager = LMAgent(supervisor="supervisor", role="grounding_manager", persona=grounding_manager_persona, vlm=vlm)
    box_checker = LMAgent(supervisor="grounding_manager", role="box_checker", persona=box_checker_persona, vlm=vlm)
    box_mover = LMAgent(supervisor="grounding_manager", role="box_mover", persona=box_mover_persona, vlm=vlm)

    system_memory = {}
    attempts = 0

    terminate_run = False

    while not terminate_run:

        write_to_log(text_log, "NUMBER OF ATTEMPTS", str(attempts))

        create_plan_prompt = supervisor_create_plan.format(**{"task": text_prompt, "image_number": image_number})

        if attempts > 0:
            replan_prompt = replan.format(**{"system_memory": system_memory})
            create_plan_prompt = replan_prompt + create_plan_prompt

        system_memory = {}

        supervisor.receive_communication(create_plan_prompt, image_list=[base64_top])
        response = supervisor.process_task()
        write_to_log(text_log, "Supervisor Create Plan", response)
        
        context = response["context"]
        high_level_plan = response["high_level_plan"]
        reasoning_process = response["reasoning_process"]

        update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                    "agent_name": supervisor.role,
                                                                    "response": response,
                                                                    "context": context})
        memory_agent.receive_communication(update_memory_prompt)
        response = memory_agent.process_task()
        write_to_log(text_log, "Memory Agent Receive Info", response)
        system_memory = update_system_memory(response, system_memory)
        write_to_log(text_log, "Memory Agent Update Memory", system_memory)

        checked_list = []
        count = 0
        while True:
            check_goal_prompt = verification_agent_check_subgoal.format(**{"task": text_prompt, 
                                                                        "image_number": image_number, 
                                                                        "high_level_plan": high_level_plan, 
                                                                        "reasoning_process": reasoning_process,
                                                                        "checked_list": str(checked_list)})
            supervisor.send_communication(verification_agent, check_goal_prompt, image_list=image_prompt)
            response = verification_agent.process_task()
            write_to_log(text_log, "Verification Agent Check Subgoal", response)
            if response["status"] == "terminate":
                break
            context = response["context"]
            subgoal_in_question = response["subgoal_in_question"]
            clarifying_questions = response["clarifying_questions"]
            checked_list = response["checked_list"]

            update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                        "agent_name": verification_agent.role,
                                                                        "response": response,
                                                                        "context": context})
            memory_agent.receive_communication(update_memory_prompt)
            response = memory_agent.process_task()
            write_to_log(text_log, "Memory Agent Receive Info", response)
            system_memory = update_system_memory(response, system_memory)
            write_to_log(text_log, "Memory Agent Update Memory", system_memory)

            if clarifying_questions != "":
                revise_plan_prompt = supervisor_revise_plan.format(**{"task": text_prompt,
                                                                    "image_number": image_number,
                                                                    "high_level_plan": high_level_plan,
                                                                    "reasoning_process": reasoning_process,
                                                                    "subgoal_in_question": subgoal_in_question,
                                                                    "clarifying_questions": clarifying_questions})
                supervisor.receive_communication(revise_plan_prompt, image_list=image_prompt)
                response = supervisor.process_task()
                write_to_log(text_log, "Supervisor Revise Plan", response)
                context = response["context"]
                high_level_plan = response["revised_high_level_plan"]
                reasoning_process = response["updated_reasoning_process"]

                update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                            "agent_name": supervisor.role,
                                                                            "response": response,
                                                                            "context": context})
                memory_agent.receive_communication(update_memory_prompt)
                response = memory_agent.process_task()
                write_to_log(text_log, "Memory Agent Receive Info", response)
                system_memory = update_system_memory(response, system_memory)
                write_to_log(text_log, "Memory Agent Update Memory", system_memory)

            count += 1
            if count == 10:
                break

        target_list = []
        action_list = []
        bbox_list = []
        write_to_log(text_log, "Verified High Level Plan", high_level_plan)
        for subgoal in high_level_plan:
            extract_targets_prompt = verification_agent_extract_targets.format(**{"subgoal": subgoal,
                                                                                "target_list": str(target_list),
                                                                                "task": text_prompt,
                                                                                "image_number": image_number})
            supervisor.send_communication(verification_agent, extract_targets_prompt, image_prompt)
            response = verification_agent.process_task()
            write_to_log(text_log, "Verification Agent Extract Targets", response)
            target_list = response["target_list"]
            context = response["context"]

            update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                        "agent_name": verification_agent.role,
                                                                        "response": response,
                                                                        "context": context})
            memory_agent.receive_communication(update_memory_prompt)
            response = memory_agent.process_task()
            write_to_log(text_log, "Memory Agent Receive Info", response)
            system_memory = update_system_memory(response, system_memory)
            write_to_log(text_log, "Memory Agent Update Memory", system_memory)

        write_to_log(text_log, "Complete Target List", str(target_list))

        if isinstance(target_list, str):
            target_list = eval(target_list)

        for target_info in target_list:
            target_type = target_info["type"]
            target_image = target_info["image"]
            target = target_info["description"]

            if target_type == "object":
                done = False
                if target_image == "env_img":
                    image_list = [base64_top]
                else:
                    img_index = int(target_image)
                    image_list = [image_prompt[img_index]]

                initial_center_list = []
                center_coord_list = []
                for choice in range(3):
                    identify_initial_center_prompt = grounding_manager_identify_initial_center.format(**{"high_level_plan": high_level_plan,
                                                                                                        "target": target})
                    supervisor.send_communication(grounding_manager, identify_initial_center_prompt, image_list)
                    response = grounding_manager.process_task(remember=False)
                    write_to_log(text_log, "Grounding Manager Identify Object Initial Location", response)
                    x = response["initial_center"]["x"]
                    y = response["initial_center"]["y"]
                    center = [x, y]

                    pil_center = draw_center(rgb_array, center)
                    save_pil(pil_center, dir=image_dir, file_name=f"manager_initial_center_{choice}")
                    initial_center_list.append(pil_image_to_base64(pil_center))
                    center_coord_list.append(center)

                select_initial_center_prompt = grounding_manager_select_initial_center.format(**{"target": target})
                supervisor.send_communication(grounding_manager, select_initial_center_prompt, image_list=initial_center_list)
                response = grounding_manager.process_task()
                write_to_log(text_log, "Grounding Manager Select Object Initial Location", response)

                center = center_coord_list[int(response["best_center_point"])]
                pil_center = draw_center(rgb_array, center)
                save_pil(pil_center, dir=image_dir, file_name="manager_best_initial_center")

                identify_initial_bbox_prompt = grounding_manager_identify_initial_bbox.format(**{"center": center,
                                                                                                "target": target})
                supervisor.send_communication(grounding_manager, identify_initial_bbox_prompt, image_list=[pil_image_to_base64(pil_center)])
                response = grounding_manager.process_task()
                write_to_log(text_log, "Grounding Manager Identify Object Initial Bounding Box", response)

                x = response["center"]["x"]
                y = response["center"]["y"]
                w = response["bounding_box"]["size"]["w"]
                h = response["bounding_box"]["size"]["h"]
                new_box = [x, y, w, h]
                box = [x, y, w, h]

                pil_box = draw_box_zoomed(rgb_array, new_box)

                box_count = 0
                while True:
                    new_pil_box = draw_box_zoomed(rgb_array, new_box, prev_box=box)
                    concat_pil_box = concat_pil_images([pil_box, new_pil_box], caption_list=["Before Revision", "After Revision"])
                    check_revision_prompt = box_checker_check_revision.format(**{"target": target})
                    grounding_manager.send_communication(box_checker, check_revision_prompt, image_list=[pil_image_to_base64(concat_pil_box)])
                    response = box_checker.process_task(remember=False)
                    write_to_log(text_log, "Box Checker Check Bounding Box", response)

                    decision = response["decision"]
                    new_pil_box = draw_box_zoomed(rgb_array, new_box)
                    save_pil(new_pil_box, dir=image_dir, file_name=f"mover_proposal")
                    save_pil(concat_pil_box, dir=image_dir, file_name=f"checker_{decision}")

                    if decision == "Accept" or done:
                        box = new_box
                        break
                    elif decision == "Revision Needed":
                        box = new_box
                        pil_box = draw_box_zoomed(rgb_array, box)

                    move_box_positon_prompt = box_mover_move_box_positon.format(**{"target": target})
                    grounding_manager.send_communication(box_mover, move_box_positon_prompt, image_list=[pil_image_to_base64(pil_box)])
                    response = box_mover.process_task()
                    write_to_log(text_log, "Box Mover Move Bounding Box", response)

                    if response["vertical_change_amount"] != "none" or response["horizontal_change_amount"] != "none":
                        new_box = adjust_box_position(box, response)

                        new_pil_box = draw_box_zoomed(rgb_array, new_box, prev_box=box)
                        concat_pil_box = concat_pil_images([pil_box, new_pil_box], caption_list=["Before Revision", "After Revision"])
                        check_revision_prompt = box_checker_check_revision.format(**{"target": target})
                        grounding_manager.send_communication(box_checker, check_revision_prompt, image_list=[pil_image_to_base64(concat_pil_box)])
                        response = box_checker.process_task()
                        write_to_log(text_log, "Box Checker Check Bounding Box", response)

                        decision = response["decision"]
                        new_pil_box = draw_box_zoomed(rgb_array, new_box)
                        save_pil(new_pil_box, dir=image_dir, file_name=f"mover_proposal")
                        save_pil(concat_pil_box, dir=image_dir, file_name=f"checker_{decision}")

                        if decision == "Accept" or done:
                            box = new_box
                            break
                        elif decision == "Revision Needed":
                            box = new_box
                            pil_box = draw_box_zoomed(rgb_array, box)
                    
                    pil_margin_inside = get_box_margin(rgb_array, box, direction="inside")
                    pil_margin_outside = get_box_margin(rgb_array, box, direction="outside")
                    concat_pil_margin = concat_pil_images([pil_margin_inside, pil_margin_outside], caption_list=["Inside Margin", "Outside Margin"])
                    adjust_margin_prompt = box_mover_adjust_margin.format(**{"target": target})
                    save_pil(concat_pil_margin, dir=image_dir, file_name="box_margin")
                    grounding_manager.send_communication(box_mover, adjust_margin_prompt, image_list=[pil_image_to_base64(concat_pil_margin)])
                    response = box_mover.process_task()
                    write_to_log(text_log, "Box Mover Adjust Margin", response)

                    new_box, done = adjust_margin(box, response)
                    new_pil_box = draw_box_zoomed(rgb_array, new_box)

                    box_count += 1
                    if box_count == 5:
                        image_list = [base64_top]
                        initial_center_list = []
                        center_coord_list = []
                        for choice in range(3):
                            identify_initial_center_prompt = grounding_manager_identify_initial_center.format(**{"high_level_plan": high_level_plan,
                                                                                                                "target": target})
                            supervisor.send_communication(grounding_manager, identify_initial_center_prompt, image_list)
                            response = grounding_manager.process_task(remember=False)
                            write_to_log(text_log, "Grounding Manager Identify Object Initial Location", response)
                            x = response["initial_center"]["x"]
                            y = response["initial_center"]["y"]
                            center = [x, y]

                            pil_center = draw_center(rgb_array, center)
                            save_pil(pil_center, dir=image_dir, file_name=f"manager_initial_center_{choice}")
                            initial_center_list.append(pil_image_to_base64(pil_center))
                            center_coord_list.append(center)

                        select_initial_center_prompt = grounding_manager_select_initial_center.format(**{"target": target})
                        supervisor.send_communication(grounding_manager, select_initial_center_prompt, image_list=initial_center_list)
                        response = grounding_manager.process_task()
                        write_to_log(text_log, "Grounding Manager Select Object Initial Location", response)

                        center = center_coord_list[int(response["best_center_point"])]

                        identify_initial_bbox_prompt = grounding_manager_identify_initial_bbox.format(**{"center": center,
                                                                                                        "target": target})
                        supervisor.send_communication(grounding_manager, identify_initial_bbox_prompt, image_list=[pil_image_to_base64(pil_center)])
                        response = grounding_manager.process_task()
                        write_to_log(text_log, "Grounding Manager Identify Object Initial Bounding Box", response)

                        x = response["center"]["x"]
                        y = response["center"]["y"]
                        w = response["bounding_box"]["size"]["w"]
                        h = response["bounding_box"]["size"]["h"]
                        new_box = [x, y, w, h]
                        box = [x, y, w, h]

                        pil_box = draw_box_zoomed(rgb_array, new_box)

                        box_count = 0
                
                bbox_list.append({"target": target, "bounding_box (x, y, w, h)": box})

                for i in range(2):
                    box = list(box)
                    pil_box, points_dict = draw_point_zoomed(rgb_array, box)
                    save_pil(pil_box, dir=image_dir, file_name=f"five_action_points_{i}")
                    select_bset_object_point_prompt = grounding_manager_select_bset_object_point.format(**{"high_level_plan": high_level_plan,
                                                                                                                "target": target,
                                                                                                                "points_dict": str(points_dict)})
                    supervisor.send_communication(grounding_manager, select_bset_object_point_prompt, image_list=[pil_image_to_base64(pil_box)])
                    response = grounding_manager.process_task()
                    write_to_log(text_log, "Grounding Manager Select Action Point", response)
                    write_to_log(text_log, "Points Dictionary", points_dict)
                    context = response["context"]
                    point_index = response["best_action_point_index"]
                    point = points_dict.get(point_index, [box[0], box[1]])
                    action_list.append({"target": target, "actionable_point": point})
                    pil_action_point = draw_center(rgb_array, point)
                    save_pil(pil_action_point, dir=image_dir, file_name=f"object_action_point_{i}")
                    box[0] = int(point[0])
                    box[1] = int(point[1])
                    box[2] = int(box[2] * 0.5)
                    box[3] = int(box[3] * 0.5)

                update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                            "agent_name": grounding_manager.role,
                                                                            "response": response,
                                                                            "context": context})
                memory_agent.receive_communication(update_memory_prompt)
                response = memory_agent.process_task()
                write_to_log(text_log, "Memory Agent Receive Info", response)
                system_memory = update_system_memory(response, system_memory)
                write_to_log(text_log, "Memory Agent Update Memory", system_memory)


            elif target_type == "location":
                if target_image == "env_img":
                    image_list = [base64_top]
                else:
                    img_index = int(target_image)
                    image_list = [image_prompt[img_index]]
                
                area_point_list = []
                point_coordinates = []
                for choice in range(3):
                    identify_location_action_point_prompt = grounding_manager_identify_area_point.format(**{"high_level_plan": high_level_plan,
                                                                                                            "target": target})
                    supervisor.send_communication(grounding_manager, identify_location_action_point_prompt, image_list)
                    response = grounding_manager.process_task(remember=False)
                    write_to_log(text_log, "Manager Find Area Actionable Point", response)
                    context = response["context"]
                    x = response["actionable_point"]["x"]
                    y = response["actionable_point"]["y"]

                    pil_area_action_point = draw_center(rgb_array, [x, y])
                    area_point_list.append(pil_image_to_base64(pil_area_action_point))
                    point_coordinates.append([x, y])
                    save_pil(pil_area_action_point, dir=image_dir, file_name=f"area_action_point_{choice}")

                    update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                                "agent_name": grounding_manager.role,
                                                                                "response": response,
                                                                                "context": context})
                    memory_agent.receive_communication(update_memory_prompt)
                    response = memory_agent.process_task()
                    write_to_log(text_log, "Memory Agent Receive Info", response)
                    system_memory = update_system_memory(response, system_memory)
                    write_to_log(text_log, "Memory Agent Update Memory", system_memory)
                
                select_best_area_point_prompt = grounding_manager_select_best_area_point.format(**{"high_level_plan": high_level_plan,
                                                                                                "target": target})
                supervisor.send_communication(grounding_manager, select_best_area_point_prompt, image_list=area_point_list)
                response = grounding_manager.process_task()
                write_to_log(text_log, "Manager Select Best Area Actionable Point", response)
                context = response["context"]
                best_index = int(response["best_action_point"])
                response["best_action_point"] = point_coordinates[best_index]
                pil_best_area = draw_center(rgb_array,  point_coordinates[best_index])
                save_pil(pil_best_area, dir=image_dir, file_name="manager_best_area_point")
                action_list.append({"target": target, "actionable_point": point_coordinates[best_index]})

                update_memory_prompt = memory_agent_update_memory.format(**{"system_memory": system_memory,
                                                                            "agent_name": grounding_manager.role,
                                                                            "response": response,
                                                                            "context": context})
                memory_agent.receive_communication(update_memory_prompt)
                response = memory_agent.process_task()
                write_to_log(text_log, "Memory Agent Receive Info", response)
                system_memory = update_system_memory(response, system_memory)
                write_to_log(text_log, "Memory Agent Update Memory", system_memory)

        write_to_log(text_log, "Complete Action Point List", action_list)

        system_memory["available_helper_functions"] = []
        for i in range(5):
            request_for_function_prompt = supervisor_request_function.format(**{
                "system_memory": system_memory,
                "action_list": action_list,
                "bbox_list": bbox_list
            })
            supervisor.receive_communication(request_for_function_prompt, image_list=[base64_top])
            response = supervisor.process_task()
            write_to_log(text_log, "Supervisor Request Function", response)
            request_function = response["request_function"]
            desired_output_variable = response["desired_output_variable"]

            if not request_function:
                break

            function_description = response["function_description"]
            generate_function_prompt = supervisor_generate_function.format(**{"function_description": function_description})
            supervisor.receive_communication(generate_function_prompt, image_list=[])
            response = supervisor.process_task()
            write_to_log(text_log, "Supervisor Generate Function", response)
            generated_function = response["generated_function"]

            run_function_prompt = supervisor_run_function.format(**{
                "system_memory": system_memory,
                "generated_function": generated_function,
                "desired_output_variable": desired_output_variable,
            })
            supervisor.receive_communication(run_function_prompt, image_list=[base64_top])
            response = supervisor.process_task()
            write_to_log(text_log, "Supervisor Run Function", response)
            run_function = response["run_function"]
            output = execute_code(run_function)
            if isinstance(output, str):
                execution_result = output
            elif isinstance(output, dict): 
                execution_result = output.get(desired_output_variable, None)

            write_to_log(text_log, "Function Execution Result", execution_result)

            verify_execution_prompt = supervisor_verify_execution.format(**{"python_script": run_function, 
                                                                "execution_result": execution_result})

            try:
                pil_traj = draw_func_traj(rgb_array, execution_result)
            except Exception as e:
                continue

            save_pil(pil_traj, dir=image_dir, file_name="Generated_Function_Execution_Results")
            base64_traj = pil_image_to_base64(pil_traj)
            supervisor.receive_communication(verify_execution_prompt, image_list=[base64_traj])
            response = supervisor.process_task()
            write_to_log(text_log, "Supervisor Verify Function Execution", response)
            correct_results = response["correct_results"]

            if correct_results:
                system_memory["available_helper_functions"].append({"python_script": run_function, 
                                                                "execution_result": output.get(desired_output_variable, None)})

        # get height
        height_info = []
        for target in action_list:
            height = average_height_in_radius(top_camera, target["actionable_point"][0], target["actionable_point"][1])
            height_info.append({"target": target["target"], "height": height})

        write_to_log(text_log, "Height Information", height_info)

        convert_actions_to_sequence_prompt = supervisor_convert_actions_to_sequence.format(**{"system_memory": system_memory, "text_prompt": text_prompt, "height_info": height_info})
        supervisor.receive_communication(convert_actions_to_sequence_prompt, image_list=[base64_top])
        response = supervisor.process_task()
        write_to_log(text_log, "Supervisor Action Sequence", response)
        action_sequence = response["action_sequence"]

        check_action_sequence_prompt = verification_agent_check_action_sequence.format(**{"system_memory": system_memory, "action_sequence": action_sequence})
        supervisor.send_communication(verification_agent, check_action_sequence_prompt, image_list=[base64_top])
        response = verification_agent.process_task()
        write_to_log(text_log, "Verify Action Sequence", response)
        action_sequence = response["verified_action_sequence"]

        # EXECUTE ACTION SEQUENCE HERE

        correct_input = False
        while not correct_input:
            user_input = input("Did the plan succeed? Enter True or False: ").strip().lower()
            if user_input == 'true':
                success = True
                correct_input = True
                
            elif user_input == 'false':
                success = False
                correct_input = True

            else:
                print("Invalid input. Please enter True or False.")

        if success:
            terminate_run = True
            break
        else:
            attempts += 1
            if attempts > 3:
                break
    
    top_camera.shutdown()

    thread.join()

if __name__ == "__main__":
    env, run_number, vlm, collect_log, args = get_input()
    if collect_log:
        image_dir = f'{env}_{run_number}'
    else:
        image_dir = "log"
    text_log = f'{image_dir}/response_log.txt'

    prepare_dir(image_dir, text_log)

    try:
        main(env, image_dir, text_log, vlm, args)
    except Exception as e:
        error_details = traceback.format_exc()
        write_to_log(text_log, "ERROR:", error_details)



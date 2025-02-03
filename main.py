from utils import *

def main(args):
    env = args.env
    env_type = args.env_type
    vlm = args.vlm
    seed = args.seed

    write_to_log(text_log, "ENV", env)
    write_to_log(text_log, "SEED NUMBER", str(seed))

    env_info = EnvInfoRetriever(env=env, env_type=env_type, seed=seed, save_dir=image_dir)
    env_info.set_robot_visibility(visible=False)

    llm_prompt = load_config("llm_prompt.yaml")

    task_specific_info = "None"
    if "rearrange" in env:
        task_specific_info = llm_prompt["rearrange"]
    elif "sweep" in env:
        task_specific_info = llm_prompt["sweep"]
    elif "snack" in env:
        task_specific_info = llm_prompt["snack"]
    elif "fruit" in env:
        task_specific_info = llm_prompt["fruit"]
    elif "follow_order" in env:
        task_specific_info = llm_prompt["follow_order"]
    elif "follow_motion" in env:
        task_specific_info = llm_prompt["follow_motion"]

    supervisor_persona = llm_prompt["supervisor"]["assistant_persona"]
    supervisor_persona = supervisor_persona.format(**{"task_specific_info": task_specific_info})

    memory_agent_persona = llm_prompt["memory_agent"]["assistant_persona"]
    memory_agent_persona = memory_agent_persona.format(**{"task_specific_info": task_specific_info})

    verification_agent_persona = llm_prompt["verification_agent"]["assistant_persona"]
    verification_agent_persona = verification_agent_persona.format(**{"task_specific_info": task_specific_info})
    
    grounding_manager_persona = llm_prompt["grounding_manager"]["assistant_persona"]
    grounding_manager_persona = grounding_manager_persona.format(**{"task_specific_info": task_specific_info})

    box_checker_persona = llm_prompt["box_checker"]["assistant_persona"]
    box_mover_persona = llm_prompt["box_mover"]["assistant_persona"]

    supervisor_create_plan = llm_prompt["supervisor"]["create_plan"]
    supervisor_revise_plan = llm_prompt["supervisor"]["revise_plan"]
    supervisor_convert_actions_to_sequence = llm_prompt["supervisor"]["convert_actions_to_sequence"]

    memory_agent_update_memory = llm_prompt["memory_agent"]["update_memory"]

    verification_agent_check_subgoal = llm_prompt["verification_agent"]["check_subgoal"]
    verification_agent_extract_targets = llm_prompt["verification_agent"]["extract_targets"]

    grounding_manager_identify_initial_center = llm_prompt["grounding_manager"]["identify_initial_center"]
    grounding_manager_select_initial_center = llm_prompt["grounding_manager"]["select_initial_center"]
    grounding_manager_identify_initial_bbox = llm_prompt["grounding_manager"]["identify_initial_bbox"]
    grounding_manager_identify_area_point = llm_prompt["grounding_manager"]["identify_area_point"]
    grounding_manager_select_best_area_point = llm_prompt["grounding_manager"]["select_best_area_point"]
    grounding_manager_identify_object_action_point = llm_prompt["grounding_manager"]["identify_object_action_point"]

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

        #***********************************************************#
        rgb_array = env_info.env_render('front')
        pil_image = draw_ticks('', '', '', '', rgb_array)
        save_pil(pil_image, dir=image_dir, file_name="Environment_Image_Front")
        base64_front = pil_image_to_base64(pil_image)

        rgb_array = env_info.env_render('top')
        pil_image = draw_ticks('', '', '', '', rgb_array)
        save_pil(pil_image, dir=image_dir, file_name="Environment_Image_Top")
        base64_top = pil_image_to_base64(pil_image)

        task = env_info.get_base_prompt()
        write_to_log(text_log, "Base Task Prompt", task)

        # translate visual prompt into text
        text_prompt, image_prompt = env_info.get_visual_prompt()

        image_number = len(image_prompt)

        for i, img in enumerate(image_prompt):
            img = draw_ticks('','','','', img)
            save_pil(img, dir=image_dir, file_name=f"visual_prompt_{i}")
            image_prompt[i] = pil_image_to_base64(img)

        write_to_log(text_log, "Text Prompt", text_prompt)

        image_prompt.append(base64_top)
        # image_prompt.append(base64_front)
        #***********************************************************#

        action = {
            'pose0_position': np.array([0.5, 0], dtype=np.float32),
            'pose0_rotation': np.array([0, 0, 0, 1], dtype=np.float32),
            'pose1_position': np.array([0.5, 0], dtype=np.float32),
            'pose1_rotation': np.array([0, 0, 0, 1], dtype=np.float32)
        }
        env_info.env.step(action)

        if attempts == 0:
            update_goal_location(env_info.env)

        write_to_log(text_log, "NUMBER OF ATTEMPTS", str(attempts))

        create_plan_prompt = supervisor_create_plan.format(**{"task": text_prompt, "image_number": image_number})

        if attempts > 0:
            replan_prompt = replan.format(**{"system_memory": system_memory})
            create_plan_prompt = replan_prompt + create_plan_prompt

        system_memory = {}

        supervisor.receive_communication(create_plan_prompt, image_list=image_prompt)
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
                    response = box_checker.process_task()
                    write_to_log(text_log, "Box Checker Check Bounding Box", response)

                    decision = response["decision"]
                    new_pil_box = draw_box_zoomed(rgb_array, new_box)
                    save_pil(new_pil_box, dir=image_dir, file_name=f"mover_proposal")
                    save_pil(concat_pil_box, dir=image_dir, file_name=f"checker_{decision}")

                    if decision == "Accept" or done:
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
                
                pil_box = draw_box_zoomed(rgb_array, box, padding=10)
                identify_object_action_point_prompt = grounding_manager_identify_object_action_point.format(**{"high_level_plan": high_level_plan,
                                                                                                            "target": target,
                                                                                                            "bounding_box": box})
                supervisor.send_communication(grounding_manager, identify_object_action_point_prompt, image_list=[pil_image_to_base64(pil_box)])
                response = grounding_manager.process_task()
                write_to_log(text_log, "Grounding Manager Action Point", response)
                context = response["context"]
                if response["actionable_point"] == "center":
                    x = box[0]
                    y = box[1]
                else:
                    x = response["actionable_point"]["x"]
                    y = response["actionable_point"]["y"]
                action_list.append({"target": target, "actionable_point": [x, y]})
                pil_action_point = draw_center(rgb_array, [x, y])
                save_pil(pil_action_point, dir=image_dir, file_name="object_action_point")

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
        convert_actions_to_sequence_prompt = supervisor_convert_actions_to_sequence.format(**{"system_memory": system_memory})
        supervisor.receive_communication(convert_actions_to_sequence_prompt, image_list=[base64_top])
        response = supervisor.process_task()
        write_to_log(text_log, "Supervisor Action Sequence", response)
        action_sequence = response["action_sequence"]

        env_info.set_robot_visibility(visible=True)

        for action_pair in action_sequence:
            action_1 = (action_pair.get("pick") or action_pair.get("start"))
            action_1 = env_info.convert_to_action(action_1)
            action_2 = (action_pair.get("place") or action_pair.get("end"))
            action_2 = env_info.convert_to_action(action_2) 
            rotation = degrees_to_quaternion(action_pair.get("rotation", 0))       
            result = env_info.execute_action(action_1, action_2, rotation)

        env_info.set_robot_visibility(visible=False)
        
        write_to_log(text_log, "FINAL RESULT", result)

        if result == "Task Succeeded.":
            terminate_run = True
            break
        else:
            attempts += 1
            if attempts > 3:
                break

if __name__ == "__main__":
    args = get_input()
    if args.collect_log:
        image_dir = f'{args.env}_{args.run_number}'
    else:
        image_dir = "log"
    text_log = f'{image_dir}/response_log.txt'

    prepare_dir(image_dir, text_log)

    try:
        main(args)
    except Exception as e:
        error_details = traceback.format_exc()
        write_to_log(text_log, "ERROR:", error_details)



# -*- coding: utf-8 -*-

# Defines the structure of MI dialogue frames using pydantic

import pprint
import json
from pydantic import BaseModel, Field

# ------------------------------------- MI Frames -------------------------------------
class GoalAndIdealFrame(BaseModel):
    """Frame for managing goals and ideals"""
    frame_type: str = "goal_and_ideal"
    content: list[str] = Field(default=[], description="content of goal and ideal")

class ProblemAndTroubleFrame(BaseModel):
    frame_type: str = Field(default="problem_and_trouble")
    type_instance_index: int
    content: list[str] = []
    detail: list[str] = []
    harm_effect: list[str] = []
    necessity_to_improve: list[str] = []

class ExperienceFrame(BaseModel):
    frame_type: str = Field(default="experience")
    type_instance_index: int
    link_frame_type_and_index: list[str] = []
    content: list[str] = []
    detail: list[str] = []
    effect: list[str] = []

class ImprovementPlanFrame(BaseModel):
    frame_type: str = Field(default="improvement_plan")
    type_instance_index: int
    link_frame_type_and_index: list[str] = []
    content: list[str] = []
    detail: list[str] = []
    confidence_to_achive: list[str] = []

class MI_Structures(BaseModel):
    """Frame structure extracted from dialogue
        usage:
            _goal = GoalAndIdealFrame()
            MI_Structures(goal_frame=_goal)
            FrameSchema = MI_Structures.model_json_schema()
            
    """
    goal_frame: GoalAndIdealFrame   # Required Goal Frame
    problem_and_trouble_frames: list[ProblemAndTroubleFrame] = []
    problem_and_trouble_links: list[str] = []
    experience_frames: list[ExperienceFrame] = []
    improvement_plan_frames: list[ImprovementPlanFrame] = []

# ------------------------------------- Tests -------------------------------------
def check_schema():
    goal = GoalAndIdealFrame()
    mi_structs_obj = MI_Structures(goal_frame=goal)

    mi_structs_schema = mi_structs_obj.model_json_schema()
    pprint.pprint(mi_structs_schema)
    print(f"title: {mi_structs_schema['title']}")
    print(f"description: {mi_structs_schema['description']}")
    print(f"parameters: {mi_structs_schema}")


if __name__ == "__main__":
    # test
    check_schema()

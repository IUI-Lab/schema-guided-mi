# -*- coding: utf-8 -*-

import pprint
import json
from typing import Optional, Union, Literal
from pydantic import BaseModel

# ------------------------------------- Schema -------------------------------------
class Foucses(BaseModel):
    frame_type: str
    instance_index: Optional[int] # 1,2,... , goal_and_ideal->None
    instance_attribute: Optional[Union[str, Literal['all']]]   # content, detail, all, None

class CounselorPolicy(BaseModel):
    intent: str
    focuses: list[Foucses]
    seek_frame_type: Optional[str]
    seek_attribute: Optional[str]
    

class CounselorPolicyWithoutIntent(BaseModel):
    focuses: list[Foucses]
    seek_frame_type: Optional[str]
    seek_attribute: Optional[str]
    
# ----- The generated detailed policy does not include intent (GT policy is used).
class CounselorStrategyRule(BaseModel):
    client_status: str
    dialogue_progress: str
    counselor_detailed_policy: CounselorPolicyWithoutIntent
    strategy_rule: str



# ------------------------------------- Tests -------------------------------------
def check_schema():
    jsonable_d = CounselorStrategyRule.model_json_schema()
    print(json.dumps(jsonable_d, indent=2))
    # json_schema = CoPolicy.model_json_schema()
    # pprint.pprint(json_schema)

if __name__ == "__main__":
    check_schema()
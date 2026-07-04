# -*- coding: utf-8 -*-

"""
Example JSON client for the Dialogue-State Similarity Search Server
(see find_near_DS_flask_server.py).

Sends a free-text query to the `/get_similar` endpoint and prints the
top-N most similar dialogue-state instances.

uv run python -m src.find_near_state.find_near_DS_json_client \
    --host 127.0.0.1 \
    --port 10800
    --top_n 5
"""

import argparse
import sys
import logging, logging.config
import pprint
import json

import requests


class FindSimilarDSClient(object):
    def __init__(self, host_port_url):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)  # suppress debug-level logs by default
        self.base_url = host_port_url

    def getSimilar(self, query_str, top_n=5):
        """
        Returns:
            dict:
                {'similar_ret': [{'annotation':{ }, 'matrix_idx': int,'similarity_score': float},
                                    ... ]}
        """
        _params = {'query_str': query_str, 'top_n': top_n}
        res = requests.post(url=self.base_url + '/get_similar',
                             json=json.dumps(_params))
        self.log.debug('res: {}'.format(res))
        self.log.debug('res.url: {}'.format(res.url))
        if res.status_code == requests.codes.ok:  # 200
            res_json = res.json()  # -> json obj
            self.log.debug('res: {}'.format(res_json))
            return res_json
        else:
            return -1

# s98
SAMPLE_QUERY = """### History
Client: Um, sometimes I drink, um, you know, maybe one or two, and then I- and then I won't drink for a while, you know. So, it's not really a-- There's not really a pattern-
Counselor: Uh-huh
Client: -you know, it's not really like a uh- it's not a habit, you know,
Counselor: Right.
Client: I-I don't know if I would consider myself alcoholic. You know, I just, uh, um, you know, again, it was poor judgment.

### Dialogue_content
{
"goal_and_ideal_frame": {"frame_type": "goal_and_ideal",
"content": []},
"problem_and_trouble_frames": [{
"frame_type": "problem_and_trouble",
"type_instnace_index": 1,
"content": ["Driving under the influence", "DUI"]
"detail": ["When I drove under the influence, I also committed a speeding violation."],
"harm_effect": [],
"necessity_to_improve": []
},
{
"frame_type": "problem_and_trouble",
"type_instnace_index": 2,
"content": ["Drinking"]
"detail": ["The last time I drank was the night I got a DUI.", "Normally I drink one or two","I’m not sure if I’m an alcoholic"],
"harm_effect": [],
"necessity_to_improve": []
}],
"problem_and_trouble_links": ["problem_and_trouble-1&2"],
"experience_frames": [{
"frame_type": "experience",
"type_instnace_index": 1,
"link_frame_type_and_index": ["problem_and_trouble-1", "problem_and_trouble-2"],
"content": ["DUI arrest"],
"detail": ["Blood alcohol level at arrest was 2.8.", "Had too much to drink at a friend’s birthday party.", "Had 5 or 6 drinks", "it was poor judgment"],
"effect": []
}],
"improvement_plan": []
}
"""


def main():
    parser = argparse.ArgumentParser(description="Example client for the dialogue-state similarity search server.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=10800, help="Server port (default: 10800)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar results to request (default: 5)")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # verbose logging for this example script
    # logging.config.fileConfig(SRC_PATH+"/logging.conf")

    client = FindSimilarDSClient(url)
    res = client.getSimilar(SAMPLE_QUERY, args.top_n)
    pprint.pprint(res)


def test_route():
    # Sanity-check that the server's root endpoint is reachable.
    parser = argparse.ArgumentParser(description="Check that the server is reachable.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=10800, help="Server port (default: 10800)")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    res = requests.get(url)
    print('res.url', res.url)
    print(res)
    print(res.status_code)
    # print(res.headers)
    # print(res.text)


if __name__ == "__main__":
    # test_route()
    main()
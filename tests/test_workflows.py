"""Offline regression checks for published workflow wiring (stdlib only).

Run: python -I tests/test_workflows.py
These check graph contracts, not model output quality. Small coherent mutations
prove that the original conditioning leak and soundtrack/model regressions fail.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


WORKFLOWS = Path(__file__).resolve().parents[1] / "example_workflows"


class Graph:
    def __init__(self, workflow):
        self.workflow = workflow
        self.nodes = {node["id"]: node for node in workflow["nodes"]}
        self.links = {link[0]: link for link in workflow["links"]}
        assert len(self.nodes) == len(workflow["nodes"]), "duplicate node ID"
        assert len(self.links) == len(workflow["links"]), "duplicate link ID"

    def of_type(self, kind):
        return [node for node in self.nodes.values() if node["type"] == kind]

    def one(self, kind):
        nodes = self.of_type(kind)
        assert len(nodes) == 1, f"expected one {kind}, found {len(nodes)}"
        return nodes[0]

    @staticmethod
    def socket(node, name):
        sockets = [item for item in node.get("inputs", []) if item["name"] == name]
        assert len(sockets) == 1, f"missing/ambiguous {node['type']}.{name}"
        return sockets[0]

    def source(self, node, name):
        link_id = self.socket(node, name)["link"]
        assert link_id is not None, f"missing connection: {node['type']}.{name}"
        link = self.links[link_id]
        return self.nodes[link[1]], link[2]

    def source_is(self, node, name, source, output_name):
        found, slot = self.source(node, name)
        return found["id"] == source["id"] and found["outputs"][slot]["name"] == output_name

    def ancestors(self, node):
        found = {}
        def visit(current):
            if current["id"] in found:
                return
            found[current["id"]] = current
            for socket in current.get("inputs", []):
                if socket.get("link") is not None:
                    visit(self.source(current, socket["name"])[0])
        visit(node)
        return list(found.values())

    def verify_links(self):
        for link_id, source_id, source_slot, target_id, target_slot, kind in self.links.values():
            assert source_id in self.nodes and target_id in self.nodes, "link endpoint missing"
            source, target = self.nodes[source_id], self.nodes[target_id]
            assert 0 <= source_slot < len(source.get("outputs", [])), "source slot out of range"
            assert 0 <= target_slot < len(target.get("inputs", [])), "target slot out of range"
            out, inp = source["outputs"][source_slot], target["inputs"][target_slot]
            assert (out.get("links") or []).count(link_id) == 1, "source link list disagrees"
            assert inp.get("link") == link_id, "target input disagrees"
            assert kind == out["type"], "link type differs from source"
            assert "*" in (out["type"], inp["type"]) or set(out["type"].split(",")) & set(inp["type"].split(",")), \
                "incompatible socket types"
        for node in self.nodes.values():
            for slot, inp in enumerate(node.get("inputs", [])):
                if inp.get("link") is not None:
                    assert inp["link"] in self.links, "input references an absent link"
                    assert self.links[inp["link"]][3:5] == [node["id"], slot], "input link endpoint disagrees"
            for slot, out in enumerate(node.get("outputs", [])):
                for link_id in out.get("links") or []:
                    assert link_id in self.links, "output references an absent link"
                    assert self.links[link_id][1:3] == [node["id"], slot], "output link endpoint disagrees"
        visiting, done = set(), set()
        def visit(node):
            assert node["id"] not in visiting, "workflow contains a cycle"
            if node["id"] in done:
                return
            visiting.add(node["id"])
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    visit(self.source(node, inp["name"])[0])
            visiting.remove(node["id"])
            done.add(node["id"])
        for node in self.nodes.values():
            visit(node)


def sampler_stages(graph):
    up = graph.one("OpenH3RegenUpscale")
    draft, _ = graph.source(up, "latent")
    assert draft["type"] == "SamplerCustomAdvanced", "upscaler must receive the generated draft"
    refine_latent = graph.one("OpenH3RegenRefineLatent")
    candidates = [node for node in graph.of_type("SamplerCustomAdvanced")
                  if graph.source_is(node, "latent_image", refine_latent, "latent")]
    assert len(candidates) == 1, "missing/ambiguous refine sampler"
    return draft, candidates[0]


def guider(graph, sampler):
    node, _ = graph.source(sampler, "guider")
    assert node["type"] == "BasicGuider", "unexpected guider contract"
    return node


def text_only_refine(graph, refine):
    encoder, _ = graph.source(guider(graph, refine), "conditioning")
    assert encoder["type"] == "CLIPTextEncode", "refine conditioning must be separately encoded text"
    media_types = {"IMAGE", "AUDIO", "VIDEO", "LATENT", "CONDITIONING"}
    for ancestor in graph.ancestors(encoder):
        assert not any(set(inp["type"].split(",")) & media_types for inp in ancestor.get("inputs", [])), \
            "media route reaches the refine text encoder"
    return encoder


def model_chain(graph, guide):
    node, _ = graph.source(guide, "model")
    chain = []
    while True:
        assert node["id"] not in {item["id"] for item in chain}, "model chain cycle"
        chain.append(node)
        if node["type"] == "UNETLoader":
            return chain
        node, _ = graph.source(node, "model")


def check_soundtrack(graph, draft):
    audio = graph.one("VAEDecodeAudio")
    assert graph.source_is(audio, "samples", draft, "output"), "soundtrack must come from the generated draft"
    saved_latent = graph.one("OpenH3SaveRefinedLatent")
    finish = graph.one("OpenH3RegenFinish")
    assert graph.source_is(saved_latent, "audio", audio, "AUDIO"), "saved latent loses the draft soundtrack"
    assert graph.source_is(finish, "audio", audio, "AUDIO"), "final soundtrack changed"
    draft_videos, final_videos = [], []
    for video in graph.of_type("CreateVideo"):
        images, _ = graph.source(video, "images")
        if images["type"] == "VAEDecode" and graph.source_is(images, "samples", draft, "output"):
            assert graph.source_is(video, "audio", audio, "AUDIO"), "draft mux loses generated audio"
            draft_videos.append(video)
        elif images["id"] == finish["id"]:
            assert graph.source_is(video, "audio", finish, "audio"), "final mux loses finished audio"
            final_videos.append(video)
    assert len(draft_videos) == len(final_videos) == 1, "draft/final mux route missing"
    for video in draft_videos + final_videos:
        assert any(graph.source_is(save, "video", video, "VIDEO") for save in graph.of_type("SaveVideo")), \
            "muxed video has no save output"


def check_ref2va(graph):
    draft, refine = sampler_stages(graph)
    r2v = graph.one("MiniMaxH3ReferenceToVideo")
    assert graph.source_is(draft, "latent_image", r2v, "LATENT"), "draft latent is not Ref2VA"
    assert graph.source_is(guider(graph, draft), "conditioning", r2v, "positive"), "draft lacks reference conditioning"
    encoder = text_only_refine(graph, refine)
    assert not any(tag in encoder["widgets_values"][0] for tag in ("<Picture", "<Video", "<Audio")), \
        "media-free refine prompt contains unresolved reference tags"
    # Native Ref2VA widgets are prompt, width, height, length, reference sizing.
    assert tuple(r2v["widgets_values"][1:4]) == (1344, 768, 124), "Ref2VA draft geometry changed"
    refs = [inp for inp in r2v["inputs"] if inp["name"].startswith("ref_images.ref_image_") and inp["link"] is not None]
    assert refs, "no image connected through native Ref2VA autogrow inputs"
    assert any(graph.source(r2v, inp["name"])[0]["type"] == "LoadImage" for inp in refs), "reference image loader missing"
    draft_chain = model_chain(graph, guider(graph, draft))
    refine_chain = model_chain(graph, guider(graph, refine))
    draft_loader, refine_loader = draft_chain[-1], refine_chain[-1]
    draft_name = draft_loader["widgets_values"][0].replace("\\", "/").lower()
    refine_name = refine_loader["widgets_values"][0].replace("\\", "/").lower()
    assert draft_loader["id"] != refine_loader["id"] and draft_name != refine_name, "draft/refine models share a loader or file"
    assert "ref2va" in draft_name or "refdelta" in draft_name, "draft model must be Ref2VA"
    assert "fl2v" in refine_name, "refine base must be FL2VA"
    assert any(node["type"] == "OpenH3RegenLora" for node in refine_chain), "refine adapter missing"
    for node in draft_chain:
        assert node["type"] != "OpenH3RegenLora", "refiner adapter applied to draft"
        if node["type"] == "LoraLoaderModelOnly":
            lora = node["widgets_values"][0].lower()
            assert not ("fl2v" in lora and "turbo" in lora), "FL2VA Turbo applied to Ref2VA draft"
    scheduler, _ = graph.source(draft, "sigmas")
    assert scheduler["type"] == "BasicScheduler", "draft scheduler changed"
    assert graph.source(scheduler, "model")[0]["id"] == draft_chain[0]["id"], "scheduler uses a different draft model"
    check_soundtrack(graph, draft)


def check_prompt(graph):
    draft, refine = sampler_stages(graph)
    i2v = graph.one("MiniMaxH3ImageToVideo")
    encoder = text_only_refine(graph, refine)
    assert graph.source_is(draft, "latent_image", i2v, "LATENT"), "draft latent is not I2V"
    assert graph.source_is(guider(graph, draft), "conditioning", i2v, "positive"), "draft frame conditioning lost"
    prompt, prompt_slot = graph.source(i2v, "prompt")
    refine_prompt, refine_slot = graph.source(encoder, "text")
    assert prompt["id"] == refine_prompt["id"] and prompt_slot == refine_slot, "draft/refine must share one prompt string"
    assert prompt["type"] == "PrimitiveStringMultiline" and prompt["widgets_values"][0].strip(), "shared prompt is empty"
    check_soundtrack(graph, draft)


def rewire(graph, target, input_name, source=None, output_name=None):
    """Mutate a connection while keeping all three serialized link records coherent."""
    socket = graph.socket(target, input_name)
    link = graph.links[socket["link"]]
    graph.nodes[link[1]]["outputs"][link[2]]["links"].remove(link[0])
    if source is None:
        socket["link"] = None
        del graph.links[link[0]]
        graph.workflow["links"].remove(link)
    else:
        slot = next(i for i, out in enumerate(source["outputs"]) if out["name"] == output_name)
        link[1:3] = [source["id"], slot]
        links = source["outputs"][slot].get("links") or []
        source["outputs"][slot]["links"] = links + [link[0]]


def reject_mutation(workflow, check, mutate, expected):
    graph = Graph(copy.deepcopy(workflow))
    mutate(graph)
    graph.verify_links()  # semantic failures must not be caused by malformed link records
    try:
        check(graph)
    except AssertionError as error:
        assert expected in str(error), f"wrong mutation failure: {error}"
    else:
        raise AssertionError("regression was not detected: " + expected)


def test_regressions(ref2va, prompt):
    for workflow, check, condition_type in ((prompt, check_prompt, "MiniMaxH3ImageToVideo"),
                                             (ref2va, check_ref2va, "MiniMaxH3ReferenceToVideo")):
        reject_mutation(workflow, check,
            lambda g: rewire(g, guider(g, sampler_stages(g)[1]), "conditioning", g.one(condition_type), "positive"),
            "refine conditioning must be separately encoded text")
    reject_mutation(ref2va, check_ref2va,
        lambda g: rewire(g, guider(g, sampler_stages(g)[0]), "model",
                         model_chain(g, guider(g, sampler_stages(g)[1]))[-1], "MODEL"),
        "draft/refine models share a loader or file")
    reject_mutation(ref2va, check_ref2va,
        lambda g: rewire(g, g.one("OpenH3SaveRefinedLatent"), "audio"),
        "missing connection: OpenH3SaveRefinedLatent.audio")
    reject_mutation(ref2va, check_ref2va,
        lambda g: rewire(g, next(n for n in g.of_type("CreateVideo")
                                if g.source(n, "images")[0]["type"] == "OpenH3RegenFinish"), "audio"),
        "missing connection: CreateVideo.audio")
    reject_mutation(prompt, check_prompt,
        lambda g: rewire(g, g.one("CLIPTextEncode"), "text"),
        "missing connection: CLIPTextEncode.text")
    reject_mutation(ref2va, check_ref2va,
        lambda g: g.one("MiniMaxH3ReferenceToVideo")["widgets_values"].__setitem__(3, 39),
        "Ref2VA draft geometry changed")


if __name__ == "__main__":
    workflows = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in WORKFLOWS.glob("*.json")}
    for workflow in workflows.values():
        Graph(workflow).verify_links()
    ref2va = workflows["Open-H3 Regenerate 2K - Ref2VA to 2K.json"]
    prompt = workflows["Open-H3 Regenerate 2K - Prompt to 2K.json"]
    check_ref2va(Graph(ref2va))
    check_prompt(Graph(prompt))
    test_regressions(ref2va, prompt)
    print("workflows: graph coherence, conditioning/model isolation, geometry and soundtrack checks passed; 7 regressions rejected")

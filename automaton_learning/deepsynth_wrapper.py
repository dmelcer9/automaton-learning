import time
from typing import NamedTuple, Dict, Any, List, Iterable

from synth.incr_nfa_dfa import make_model, convert_to_dfa_plot
from synth.synth_wrapper import fix_states


# Wrapper for deepsynth code to work around some confusing design choices in the original
# Might be worth abstracting this

class HyperParams(NamedTuple):
    """Original deepsynth read from command-line arguments; but that won't work here"""
    window: int
    num_states: int  # This is __only__ used to initialize num_states in DeepsynthState
    target: str  # Appears to be some file path
    order: str  # bts, stb, random, or same. Default is same
    incr: bool  # Almost always true?


def get_default_hparams() -> HyperParams:
    """Default from command-line"""
    return HyperParams(window=3, num_states=2, target="models", order="same", incr=True)


class DeepsynthState(NamedTuple):
    """These objects comprise the internal state of deepsynth; packing them together for convenience.
    Really it's somewhat opaque to us"""
    num_states: int
    var: Dict
    input_dict: Dict
    hyperparams: HyperParams
    dfa_model: Any
    model_gen: Any
    iter_num: int


def wrapped_dfa_init(hyperparams: HyperParams):
    """Create a new empty internal deepsynth state"""
    len_seq = hyperparams.window

    var = {'incr': 0, 'events_tup_to_list': [], 'o_event_uniq': [], 'org_trace': [], 'seq_input_uniq': []}
    input_dict = {'event_id': [], 'seq_input_uniq': [], 'event_uniq': [], 'len_seq': len_seq}

    return DeepsynthState(hyperparams.num_states, var, input_dict, hyperparams, [], [], 0)


# The deepsynth process_dfa function always leaves a gap in the original starting state (usually state 2)
# and uses an unusual mapping between APs and transition numbers. This function fixes these issues
def nice_process_dfa(dfa_model, input_dict):
    real_start_state = [x[2] for x in dfa_model if x[0] == 1 and x[1] == 1][0]

    all_states = set(x[0] for x in dfa_model).union(set(x[2] for x in dfa_model))
    all_states.remove(1)  # The original "fake" start state
    list_of_states = list(sorted(all_states))  # All the other states

    # Start numbering the states from 0
    old_to_new_state_mapping = {old_state_num: idx for idx, old_state_num in enumerate(list_of_states)}

    # Number the non-"start" actions from 0, taking into account the fact that input_dict is 1-indexed
    action_mapping = {(action_map_idx + 1): action_num for action_map_idx, action_num in
                      enumerate(input_dict["event_uniq"]) if action_num != "start"}

    converted_dfa = [
        (old_to_new_state_mapping[start_state], action_mapping[action], old_to_new_state_mapping[end_state]) for
        start_state, action, end_state in dfa_model if start_state != 1]
    new_real_start_state = old_to_new_state_mapping[real_start_state]

    return converted_dfa, new_real_start_state


def reduce_stutter(trace: List[int]):
    """Allow only a single stutter"""
    new_trace = []
    last_stutter = None
    stutter_count = 0

    for elem in trace:
        if elem == last_stutter:
            stutter_count += 1
            if stutter_count <= 2:
                new_trace.append(elem)
        else:
            last_stutter = elem
            stutter_count = 1
            new_trace.append(elem)

    return new_trace


def remove_stutters(trace: List[int]) -> List[int]:
    """Remove stutters entirely (can use torch if needs to be faster)"""
    last_elem = None
    output = []
    for item in trace:
        if item == last_elem:
            continue
        else:
            output.append(item)
            last_elem = item

    return output


def remove_zeros(trace: List[int]) -> List[int]:
    return [t for t in trace if t != 0]


# This is nearly a direct clone of deepsynth/synth.synth_wrapper.dfa_update
# Only difference being that some weird "start" string manipulation is gone, and some unnecessary arguments are removed
def wrapped_dfa_update(traces: Iterable[List[int]], state: DeepsynthState):
    traces = [remove_zeros(t) for t in traces]

    num_states, var, input_dict, hyperparams, dfa_model, model_gen, iter_num = state

    start_time = time.time()

    old_dfa_model = dfa_model.copy()

    # TODO try to dig into deepsynth code itself to remove dependence on "start" string
    # noinspection PyTypeChecker
    traces_as_str_with_start = [["start"] + trace + ["start"] for trace in traces]

    # Use set to deduplicate the event list
    unique_event_tuples = list(set(tuple(x) for x in traces_as_str_with_start))

    # But now this is a list of tuples, we want a list of lists
    unique_event_lists = [list(x) for x in unique_event_tuples]

    var['events_tup_to_list'] = unique_event_lists

    for event_list in unique_event_lists:
        var['org_trace'].extend(event_list)  # TODO try org_trace by itself in a loop, and make_model only called once
        model_gen, var, input_dict, num_states = make_model(event_list, model_gen, var, hyperparams, num_states,
                                                            input_dict, start_time)

    nfa_model = model_gen.copy()
    # full_events is never used in this function, so we don't need to keep it around
    dfa_model, var, input_dict, num_states = convert_to_dfa_plot(None, model_gen, input_dict, num_states,
                                                                 hyperparams, var, start_time, iter_num)
    model_gen = nfa_model.copy()

    # I believe that this correlates old states with new states
    dfa_model = fix_states(old_dfa_model, dfa_model, input_dict['event_id'], num_states)

    # processed_dfa isn't used at all by deepsynth, so we can make it as convenient to use as we want
    processed_dfa, start_state = nice_process_dfa(dfa_model, input_dict)

    return processed_dfa, start_state, DeepsynthState(num_states, var, input_dict, hyperparams, dfa_model, model_gen,
                                                      iter_num + 1)

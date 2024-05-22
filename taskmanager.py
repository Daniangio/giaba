import numpy as np
import pandas as pd
import random
import tqdm
from typing import List
from itertools import permutations
from utils import *

import warnings 
warnings.filterwarnings('ignore')


class Task:

    def __init__(
        self,
        id: int,
        type: int,
        length: int,
        deadline: int,
        oh: int,
        penalty: int,
    ) -> None:
        self.id = id
        self.type = type
        self.length = length
        self.deadline = deadline
        self.oh = oh
        self.penalty = penalty


class Result:

    @staticmethod
    def join(results):
        return Result(
            None,
            np.concatenate([r.options for r in results], axis=0),
            np.concatenate([r.penalty for r in results], axis=0),
            np.concatenate([r.task_length for r in results], axis=0),
            np.concatenate([r.M for r in results], axis=0),
        )

    def __init__(
        self,
        best_options_indices,
        options,
        penalty,
        task_length,
        M,
    ) -> None:
        if best_options_indices is None:
            best_options_indices = np.arange(len(options))
        self.len = len(best_options_indices)
        self.options = options[best_options_indices]
        self.penalty = penalty[best_options_indices]
        self.task_length = task_length[best_options_indices]
        self.M = M[best_options_indices]
    
    @property
    def cumlength(self):
        return np.sum(self.task_length, axis=-1)
    
    @property
    def cumpenalty(self):
        return np.sum(self.penalty, axis=-1)
    
    @property
    def task_indices(self):
        return self.options[..., 5]
    
    def __len__(self):
        return self.len
    
    def __eq__(self, other):
        for task_indices_row in self.task_indices:
            found = False
            for other_task_indices_row in other.task_indices:
                if np.all(task_indices_row == other_task_indices_row):
                    found = True
                    break
            if not found:
                return False
        return True


class Results:

    def __init__(self) -> None:
        self.best_result: Result = None
    
    def append(self, result: Result):
        if self.best_result is None:
            self.best_result = result
            return
        
        results = np.array([self.best_result, result])

        best_sorted_solution_index = np.lexsort(
            (
                [r.cumlength[0] for r in results],
                [r.cumpenalty[0] for r in results],
            )
        )[0]

        self.best_result = results[best_sorted_solution_index]
    
    def __len__(self):
        return len(self.best_result)

    @property
    def penalty(self):
        return self.best_result.penalty
    
    @property
    def cumpenalty(self):
        return self.best_result.cumpenalty
    
    @property
    def task_length(self):
        return self.best_result.task_length

    @property
    def cumlength(self):
        return self.best_result.cumlength
    
    @property
    def task_indices(self):
        return self.best_result.task_indices


class TaskManager:

    def __init__(self) -> None:
        self.types = []
        self.lengths = []
        self.deadlines = []
        self.ohs = []
        self.penalties = []
        self.ids = []
        self.tasks: List[Task] = []
        self.task_board = None
        self.heuristics = []
        self.id_counter: int = 0
    
    def add_task(
        self,
        type: int,
        length: int,
        deadline: int,
        oh: int,
        penalty: int,
    ):
        task = Task(
            id=self.id_counter,
            type=type,
            length=length,
            deadline=deadline,
            oh=oh,
            penalty=penalty,
        )
        self.tasks.append(task)
        self.id_counter += 1
    
    def add_random_task(
        self,
        type_range = (0, 5),
        deadline_range = (10, 365),
        length_range = (1, 5),
        oh_range = (1, 3),
        penalty_range = (1, 3),
    ):
        type = random.randint(*type_range)
        deadline = random.randint(*deadline_range)
        length = random.randint(*length_range)
        oh = random.randint(*oh_range)
        penalty = random.randint(*penalty_range)

        self.add_task(type, length, deadline, oh, penalty)

    def complete(self):
        for task in self.tasks:
            self.types.append(task.type)
            self.lengths.append(task.length)
            self.deadlines.append(task.deadline)
            self.ohs.append(task.oh)
            self.penalties.append(task.penalty)
            self.ids.append(task.id)
        self.types = np.array(self.types)
        self.lengths = np.array(self.lengths)
        self.deadlines = np.array(self.deadlines)
        self.ohs = np.array(self.ohs)
        self.penalties = np.array(self.penalties)
        self.ids = np.array(self.ids)

        self.task_board: pd.DataFrame = pd.DataFrame({
            'type': self.types,
            'length': self.lengths,
            'deadline': self.deadlines,
            'oh': self.ohs,
            'penalty': self.penalties,
            'id': self.ids,
        })

        deadline_heuristic = self.task_board.sort_values(by=['deadline', 'type']).reset_index(drop=True)
        difference_heuristic = self.task_board.assign(
                difference=lambda df: df['deadline'] - df['length']
            ).sort_values(by=['difference', 'type']).reset_index(drop=True)
        
        self.heuristics = [
            deadline_heuristic,
            difference_heuristic,
        ]
    
    def save_tasks(self, filename: str):
        if self.task_board is None:
            raise Exception('No tasks are present. Maybe you missed to call the "complete()" method before?')
        self.task_board.to_csv(filename)
    
    def load_tasks(self, filename: str):
        task_board = pd.read_csv(filename, index_col=0)
        for index, row in task_board.iterrows():
            self.add_task(*row[:-1])
    
    def solve(self, chunk_size: int = 5):
        results = Results()
        print("Iterating initial heuristics...")
        for heuristic in tqdm.tqdm(self.heuristics):
            solution = None
            start_index = 0
            chunk_dfs = split_reminder(heuristic, chunk_size)
            print("Solving and refining chunks...")
            for chunk_df in tqdm.tqdm(chunk_dfs):
                chunk = chunk_df.values
                num_chunk_tasks = chunk.shape[-2]
                
                perm_indices = np.stack(list(permutations(np.arange(num_chunk_tasks))), axis=0)
                batch_options = chunk[perm_indices][np.newaxis, ...]

                is_improving = True
                local_results = [] # local suboptimal could be optimal globally
                while is_improving:
                    result = self.solve_chunk(batch_options, solution)
                    refined_result = self.refine_penalty(result, start_index=start_index)
                    refined_result = self.refine_task_types(refined_result)
                    local_results.append(refined_result)
                    solution = refined_result.options
                    if len(solution.shape) == 2:
                        solution = solution[np.newaxis, ...]
                    if refined_result == result:
                        is_improving = False
                    else:
                        chunk = np.copy(solution[:, -num_chunk_tasks:])[:, perm_indices]
                        solution = np.copy(solution[:, :-num_chunk_tasks])
                result = Result.join(local_results)
                solution = result.options
                start_index += len(chunk)

            print("Optimizing consecutive task types...")
            result = self.refine_penalty(result, start_index=start_index)
            # result = self.refine_task_types(result)
            results.append(result)
        
        return results
    
    def solve_chunk(self, batch_options: np.ndarray, solution: np.ndarray) -> Result:
        if solution is not None:
            combinations = []
            for options in batch_options:
                for sol_row in solution:
                    combinations.append(
                        np.concatenate((np.repeat(sol_row[np.newaxis, ...], len(options), axis=0), options), axis=-2)
                    )
            options = np.concatenate(combinations, axis=0)
        else:
            options = np.concatenate(batch_options, axis=0)
        return self.pick_best(options)

    def refine_penalty(self, result: Result, start_index: int):
        if start_index == 0:
            return result
        options = [opt for opt in result.options]
        for row_idx in range(len(result)):
            penalty_indices = np.where(result.penalty[row_idx, start_index:] > 0)[0]
            first_sequence = np.split(penalty_indices, np.where(np.diff(penalty_indices) != 1)[0] + 1)[0]
            if len(first_sequence) > 0:
                first_sequence = np.arange(start_index, start_index + first_sequence[-1] + 1)
                prev_penalty_indices = np.where(result.penalty[row_idx, :start_index] == 0)[-1] - 1
                prev_penalty_indices = prev_penalty_indices[prev_penalty_indices > 0]
                if len(prev_penalty_indices) > 0 and prev_penalty_indices[-1] == start_index - 2:
                    arr = result.options[row_idx]
                    for seq_src, seq_trg in swap_combs(first_sequence, prev_penalty_indices):
                        options.append(move_and_permute_elements(np.copy(arr), seq_src, seq_trg)[0])
        options = np.stack(options, axis=0)
        return self.pick_best(options)

    def refine_task_types(self, result: Result):
        best_options = result.options
        u_types = np.unique(best_options[..., 0])

        combinations = []
        for row in best_options:
            for type in u_types:
                new_combination = move_consecutive_types(row, type)
                if len(new_combination) > 0:
                    combinations.append(np.stack(new_combination, axis=0))
        if len(combinations) > 0:
            options = np.concatenate((best_options, np.concatenate(combinations, axis=0)), axis=0)
            new_result = self.pick_best(options)
            if new_result == result:
                return new_result                
            return self.refine_task_types(new_result)
        return self.pick_best(best_options)
    
    def pick_best(self, options: np.ndarray):
        penalty, task_length, M, penalty_score, length_score = self.evaluate(options)

        sort_indices = np.lexsort((length_score, penalty_score))
        sorted_penalty_score = penalty_score[sort_indices]
        sorted_length_score = length_score[sort_indices]

        task_type =    options[..., 0]
        task_length =  options[..., 1]
        deadline =     options[..., 2]
        task_oh =      options[..., 3]
        task_penalty = options[..., 4]

        min_penalty_indices = np.where(sorted_penalty_score == np.min(sorted_penalty_score))[0]
        min_length_indices = np.where(sorted_length_score[min_penalty_indices] == np.min(sorted_length_score[min_penalty_indices]))[0]

        passed_indices = sort_indices[min_length_indices]

        # When 3 or more tasks of the same type are in a row, order them according to > deadline, < penalty, < oh.
        passed_deadline_score = deadline[passed_indices]
        weight_matrix = np.ones_like(passed_deadline_score, dtype=np.float32)
        weight_col = (np.power(.5, np.arange(0, weight_matrix.shape[1])).astype(np.float32))
        weight_matrix *= weight_col[np.newaxis, :]
        passed_deadline_score = np.sum(passed_deadline_score * weight_matrix, axis=-1)
        passed_penalty_score = np.sum(-task_penalty[passed_indices] * weight_matrix, axis=-1)
        passed_oh_score = np.sum(-task_oh[passed_indices] * weight_matrix, axis=-1)
        ordered_passed_indices = passed_indices[np.lexsort((passed_oh_score, passed_penalty_score, passed_deadline_score))]
    
        first_occurrence_indices = first_n_occurrences(task_type[ordered_passed_indices][:, -1], 2)
        best_options_indices = passed_indices[first_occurrence_indices]
        return Result(
                best_options_indices,
                options,
                penalty,
                task_length,
                M,
            )

    def evaluate(self, options: np.ndarray):
        _options = np.copy(options)
        task_type =    _options[..., 0]
        task_length =  _options[..., 1]
        deadline =     _options[..., 2]
        task_oh =      _options[..., 3]
        task_penalty = _options[..., 4]
        oh_mask = np.zeros_like(task_type, dtype=bool)
        oh_mask[:, 1:] = task_type[:, :-1] != task_type[:, 1:]
        oh_mask[:, 0] = True
        
        task_length[oh_mask] += task_oh[oh_mask]

        M = np.cumsum(task_length, axis=-1)
        M_zeros = np.zeros_like(M)
        
        penalty = np.maximum(M_zeros, M - deadline) * task_penalty
        penalty_score = np.sum(penalty, axis=-1)
        length_score = np.sum(task_length, axis=-1)
        
        return penalty, task_length, M, penalty_score, length_score


#   	  local_results = []
#         first_n = 1
#         for best_n in range(1, first_n + 1):
#             if len(np.unique(penalty_score)) > best_n:
#                 n_min_penlties = np.partition(np.unique(penalty_score), best_n)[best_n - 1]
#             else:
#                 n_min_penlties = np.unique(penalty_score)[min(len(np.unique(penalty_score)) - 1, best_n - 1)]
#             min_penalty_indices = np.where(sorted_penalty_score == n_min_penlties)[0]
#             if len(np.unique(sorted_length_score[min_penalty_indices])) > first_n:
#                 n_min_length = np.partition(np.unique(sorted_length_score[min_penalty_indices]), first_n)[:first_n]
#             else:
#                 n_min_length = np.unique(sorted_length_score[min_penalty_indices])
#             min_length_indices = np.where(np.isin(sorted_length_score[min_penalty_indices], n_min_length))[0]

#             passed_indices = sort_indices[min_length_indices]

#             # When 3 or more tasks of the same type are in a row, order them according to > deadline, < penalty, < oh.
#             passed_deadline_score = deadline[passed_indices]
#             weight_matrix = np.ones_like(passed_deadline_score, dtype=np.float32)
#             weight_col = (np.power(.5, np.arange(0, weight_matrix.shape[1])).astype(np.float32))
#             weight_matrix *= weight_col[np.newaxis, :]
#             passed_deadline_score = np.sum(passed_deadline_score * weight_matrix, axis=-1)
#             passed_penalty_score = np.sum(-task_penalty[passed_indices] * weight_matrix, axis=-1)
#             passed_oh_score = np.sum(-task_oh[passed_indices] * weight_matrix, axis=-1)
#             ordered_passed_indices = passed_indices[np.lexsort((passed_oh_score, passed_penalty_score, passed_deadline_score))]
        
#             first_occurrence_indices = first_n_occurrences(task_type[ordered_passed_indices][:, -1], first_n)
#             best_options_indices = passed_indices[first_occurrence_indices]
#             local_results.append(
#                 Result(
#                     best_options_indices,
#                     options,
#                     penalty,
#                     task_length,
#                     M,
#                 )
#             )

#         return Result.join(local_results)
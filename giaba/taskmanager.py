import tqdm
import numpy as np
import pandas as pd
import random
from typing import List
from itertools import permutations
from giaba.utils import *

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
            np.concatenate([r.length  for r in results], axis=0),
        )

    def __init__(
        self,
        best_options_indices,
        options,
        penalty,
        length,
    ) -> None:
        if best_options_indices is None:
            best_options_indices = np.arange(len(options))
        self.len = len(best_options_indices)
        self.options = options[best_options_indices]
        self.penalty = penalty[best_options_indices]
        self.length  = length[best_options_indices]
    
    @property
    def type(self):
        return self.options[..., 0]

    @property
    def deadline(self):
        return self.options[..., 2]
    
    @property
    def cumlength(self):
        return np.cumsum(self.length, axis=-1)
    
    @property
    def sumlength(self):
        return np.sum(self.length, axis=-1)
    
    @property
    def cumpenalty(self):
        return np.cumsum(self.penalty, axis=-1)
    
    @property
    def sumpenalty(self):
        return np.sum(self.penalty, axis=-1)
    
    @property
    def task_types(self):
        return self.options[..., 0]
    
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
                [r.sumlength[0]  for r in results],
                [r.sumpenalty[0] for r in results],
            )
        )[0]

        self.best_result = results[best_sorted_solution_index]
    
    def __len__(self):
        return len(self.best_result)
    
    @property
    def length(self):
        return self.best_result.length

    @property
    def cumlength(self):
        return self.best_result.cumlength
    
    @property
    def sumlength(self):
        return self.best_result.sumlength
    
    @property
    def penalty(self):
        return self.best_result.penalty
    
    @property
    def cumpenalty(self):
        return self.best_result.cumpenalty
    
    @property
    def sumpenalty(self):
        return self.best_result.sumpenalty
    
    @property
    def task_types(self):
        return self.best_result.task_types
    
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
            ).sort_values(by=['difference', 'deadline', 'type']).reset_index(drop=True)
        
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
            batch_solutions = None
            start_index = 0
            chunk_dfs = split_reminder(heuristic, chunk_size)
            print("Solving and refining chunks...")
            for chunk_df in tqdm.tqdm(chunk_dfs):
                chunk = chunk_df.values
                num_chunk_tasks = chunk.shape[-2]
                
                perm_indices = np.stack(list(permutations(np.arange(num_chunk_tasks))), axis=0)
                batch_options = np.expand_dims(chunk[perm_indices], axis=0)

                is_improving = True
                local_results = [] # local suboptimal could be optimal globally
                while is_improving:
                    # batch_options:   (batch_size, num_perms, chunk_size, options)
                    # batch_solutions: (batch_size, num_solutions, solution_size, options)
                    result = self.solve_chunk(batch_options, batch_solutions)
                    refined_result = self.refine_task_types(result)
                    refined_result = self.refine_penalty(refined_result, start_index=start_index)
                    refined_result = self.refine_task_types(refined_result)
                    local_results.append(refined_result)
                    # is_improving = False
                    if refined_result == result:
                        is_improving = False
                    else:
                        refined_solutions = refined_result.options
                        batch_options = np.copy(refined_solutions[:, -num_chunk_tasks:])[:, perm_indices]
                        batch_solutions = np.expand_dims(np.copy(refined_solutions[:, :-num_chunk_tasks]), axis=1)
                result = Result.join(local_results)
                batch_solutions = result.options
                while batch_solutions.ndim < 4:
                    batch_solutions = np.expand_dims(batch_solutions, axis=0)
                start_index += num_chunk_tasks
            results.append(result)
        
        return results
    
    def solve_chunk(self, batch_options: np.ndarray, batch_solution: np.ndarray) -> Result:
        if batch_solution is not None:
            combinations = []
            for options, solutions in zip(batch_options, batch_solution):
                for solution in solutions:
                    combinations.append(
                        np.concatenate((np.repeat(solution[np.newaxis, ...], len(options), axis=0), options), axis=-2)
                    )
            options = np.concatenate(combinations, axis=0)
        else:
            options = np.concatenate(batch_options, axis=0)
        return self.pick_best(options)

    def refine_penalty(self, result: Result, start_index: int):
        if start_index == 0:
            return result        

        combinations = []
        for option, penalty in zip(result.options, result.penalty):
            first_penalty_index = np.where(penalty[start_index:] > 0)[0]
            if len(first_penalty_index) > 0:
                first_penalty_index = first_penalty_index[0]
                chunk_idcs = np.arange(start_index, start_index + first_penalty_index + 1)
                for shift in range(1, start_index):
                    combinations.append(shift_chunk(option, chunk_idcs, -shift))
        if len(combinations) > 0:
            options = np.concatenate([result.options, np.stack(combinations, axis=0)], axis=0)
            new_result = self.pick_best(options)
            if new_result == result:
                return new_result 
            return self.refine_penalty(new_result, start_index)
        return result

    def refine_task_types(self, result: Result):
        options = result.options
        u_types = np.unique(result.task_types)

        combinations = []
        for option in options:
            for type in u_types:
                new_combination = stack_consecutive_types(option, type)
                if len(new_combination) > 0:
                    combinations.append(np.stack(new_combination, axis=0))
        if len(combinations) > 0:
            options = np.concatenate((options, np.concatenate(combinations, axis=0)), axis=0)
            options = np.unique(options, axis=0) # drop duplicates
            new_result = self.pick_best(options)
            if new_result == result:
                return new_result                
            return self.refine_task_types(new_result)
        return result
    
    def pick_best(self, options: np.ndarray):
        penalty, length = self.evaluate(options)
        penalty_score = np.sum(penalty, axis=-1)
        length_score = np.sum(length, axis=-1)

        sort_indices = np.lexsort((length_score, penalty_score))
        sorted_penalty_score = penalty_score[sort_indices]
        sorted_length_score  = length_score[sort_indices]

        task_type     = options[..., 0]
        task_deadline = options[..., 2]
        task_oh       = options[..., 3]
        task_penalty  = options[..., 4]

        min_penalty_indices = np.where(sorted_penalty_score == np.min(sorted_penalty_score))[0]
        min_length_indices = np.where(sorted_length_score[min_penalty_indices] == np.min(sorted_length_score[min_penalty_indices]))[0]

        passed_indices = sort_indices[min_length_indices]

        # When 3 or more tasks of the same type are in a row, order them according to > deadline, < penalty, < oh.
        passed_deadline_score = task_deadline[passed_indices]
        weight_matrix = np.ones_like(passed_deadline_score, dtype=np.float32)
        weight_col = (np.power(.5, np.arange(0, weight_matrix.shape[1])).astype(np.float32))
        weight_matrix *= weight_col[np.newaxis, :]
        passed_deadline_score = np.sum(passed_deadline_score * weight_matrix, axis=-1)
        passed_penalty_score = np.sum(-task_penalty[passed_indices] * weight_matrix, axis=-1)
        passed_oh_score = np.sum(-task_oh[passed_indices] * weight_matrix, axis=-1)
        ordered_passed_indices = passed_indices[np.lexsort((passed_oh_score, passed_penalty_score, passed_deadline_score))]
    
        first_occurrence_indices = first_n_occurrences(task_type[ordered_passed_indices][:, -1], 5)
        best_options_indices = passed_indices[first_occurrence_indices]
        return Result(
            best_options_indices,
            options,
            penalty,
            length,
        )

    def evaluate(self, options: np.ndarray):
        _options = np.copy(options)
        task_type     = _options[..., 0]
        task_length   = _options[..., 1]
        task_deadline = _options[..., 2]
        task_oh       = _options[..., 3]
        task_penalty  = _options[..., 4]
        oh_mask = np.zeros_like(task_type, dtype=bool)
        oh_mask[:, 1:] = task_type[:, :-1] != task_type[:, 1:]
        oh_mask[:, 0] = True
        
        length = np.copy(task_length)
        length[oh_mask] += task_oh[oh_mask]

        M = np.cumsum(length, axis=-1)
        M_zeros = np.zeros_like(M)
        penalty = np.maximum(M_zeros, M - task_deadline) * task_penalty
        
        return penalty, length


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
#                 )
#             )

#         return Result.join(local_results)
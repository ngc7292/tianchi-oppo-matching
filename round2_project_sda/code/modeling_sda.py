# -*- coding: utf-8 -*-
"""
__title__="modeling_sda"
__author__="ngc7293"
__mtime__="2021/5/8"
"""
import torch
from copy import deepcopy


class SDAModel(torch.nn.Module):
    def __init__(self, bert_model: torch.nn.Module, avg_times=5, use_student=False,
                 lambda_distillation=1):
        super(SDAModel, self).__init__()
        assert avg_times >= 0

        self.student = bert_model
        self.avg_times = avg_times
        self.use_student = use_student
        self.lambda_distillation = lambda_distillation
        self.update_index = 0
        self.update_time = 0
        self.updated = torch.nn.Parameter(torch.zeros(max(self.avg_times, 1)), requires_grad=False)
        n_teacher = 1 if avg_times <= 1 else (avg_times + 1)
        self.teacher = torch.nn.ModuleList([deepcopy(self.student) for _ in range(n_teacher)])
        self.check = 0

    def forward(self, input_ids, token_type_ids, attention_mask=None, labels=None):
        if labels is None:
            return self.predict(input_ids, token_type_ids, attention_mask)
        else:
            loss, logits = self.student(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                        labels=labels)[:2]

            output = (logits,)

            if labels is not None:
                distillation_loss = 0
                if self.lambda_distillation != 0:
                    distillation_loss_func = torch.nn.MSELoss()
                    with torch.no_grad():
                        teacher_loss, teacher_logits = self.teacher[-1](input_ids,
                                                                        token_type_ids=token_type_ids,
                                                                        attention_mask=attention_mask,
                                                                        labels=labels)[:2]
                    distillation_loss = distillation_loss_func(logits, teacher_logits)

                loss = loss + self.lambda_distillation * distillation_loss

                output = (loss,) + output

            return output

    def predict(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            if self.use_student:
                logits = self.student(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            else:
                logits = self.teacher[-1](input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        return (logits,)

    def update_teacher(self):
        with torch.no_grad():
            self.update_time += 1
            student_param_dict = {n: p for n, p in self.student.named_parameters()}
            if self.avg_times == 0:  # average from time step 0
                teacher_dict = {n: p for n, p in self.teacher[0].named_parameters()}
                new_teacher_dict = {n: student_param_dict[n] / (self.update_time + 1) +
                                       teacher_dict[n] * self.update_time / (self.update_time + 1)
                                    for n in student_param_dict.keys()}
            elif self.avg_times == 1:  # only consider last time step
                new_teacher_dict = student_param_dict
            else:
                param_list = [n for n, p in self.student.named_parameters()]

                self.teacher[self.update_index].load_state_dict(student_param_dict)
                self.updated[self.update_index] = 1
                self.update_index = (self.update_index + 1) % self.avg_times

                updated = self.updated.sum().item()
                assert updated != 0
                new_teacher_dict = {n: 0 for n in param_list}
                for idx in range(self.avg_times):
                    if self.updated[idx] == 1:
                        teacher = {n: p for n, p in self.teacher[idx].named_parameters()}
                        for n in param_list:
                            new_teacher_dict[n] = new_teacher_dict[n] + teacher[n] / updated

            self.teacher[-1].load_state_dict(new_teacher_dict)

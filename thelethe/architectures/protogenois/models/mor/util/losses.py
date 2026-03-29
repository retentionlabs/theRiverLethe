import torch
import torch.nn.functional as F


def substract_max_logits(logits):
    # for numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits = logits - max_logits
    return logits


def forward_kl(logits, teacher_logits):
    
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
    distill_loss = torch.sum(
        teacher_probs * F.log_softmax(logits, dim=-1), dim=-1
    )
        
    distill_loss = -torch.mean(distill_loss)
    return distill_loss


def reverse_kl(logits, teacher_logits):

    student_probs = F.softmax(logits, dim=-1)
    with torch.no_grad():
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
        
    distill_loss = torch.sum(
        student_probs * teacher_logprobs
        - student_probs * F.log_softmax(logits, dim=-1), dim=-1
    )
    
    distill_loss = -torch.mean(distill_loss)
    return distill_loss


def js_divergence(logits, teacher_logits, lam=0.5):
    
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
        
    mixed_logprobs = torch.log(
        (1-lam) * teacher_probs + lam * student_probs + 1e-10
    )
    
    distill_loss = lam * -torch.mean(
        torch.sum(student_probs * mixed_logprobs - student_probs * F.log_softmax(logits, dim=-1), dim=-1)
    )
    
    distill_loss += (1-lam) * -torch.mean(
        torch.sum(teacher_probs * mixed_logprobs - teacher_probs * F.log_softmax(teacher_logits, dim=-1), dim=-1)
    )
    return distill_loss


def tv_distance(logits, teacher_logits):
    
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(logits, dim=-1)
    
    distill_loss = 0.5 * torch.norm(teacher_probs - student_probs, p=1, dim=-1)
    distill_loss = torch.mean(distill_loss)
    return distill_loss


DISTILL_LOSSES = {
    "forward_kl": forward_kl,
    "reverse_kl": reverse_kl,
    "js_divergence": js_divergence,
    "tv_distance": tv_distance,
}
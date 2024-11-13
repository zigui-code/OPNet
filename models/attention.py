import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, patch_size = 3, propagate_size = 3, stride = 1):
        super(SelfAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        
    def forward(self, foreground):
        bz, nc, w, h = foreground.size()
        background = foreground.clone()
        background = background
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i:i+1]
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim = True)**0.5
            conv_kernels = conv_kernels/norm_factor
            
            conv_result = F.conv2d(feature_map, conv_kernels, padding = self.patch_size//2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.avg_pool2d(conv_result, 3, 1, padding = 1)*9
            attention_scores = F.softmax(conv_result, dim = 1)
            
            feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride = 1, padding = self.patch_size//2)  # Note here, conv_kernels -> conv_kernels_all[i], keep amplitude information 
            final_output = feature_map
            output_tensor.append(final_output)
            att_score.append(attention_scores.permute(0,2,3,1).view(w*h,-1))  # 2D tensor, prob in dim=1

        return torch.cat(output_tensor, dim = 0), torch.cat(att_score, dim=0)
                
class AttentionModule(nn.Module):
    
    def __init__(self, inchannel, patch_size_list = [1], propagate_size_list = [3], stride_list = [1]):
        assert isinstance(patch_size_list, list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.att = SelfAttention(patch_size_list[0], propagate_size_list[0], stride_list[0])

        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size = 1)
        
    def forward(self, foreground):
        outputs, att_score = self.att(foreground)
        outputs = torch.cat([outputs, foreground],dim = 1)
        outputs = self.combiner(outputs)
        return outputs




from einops import rearrange


##  Top-K Sparse Attention (TKSA)  旨在通过仅关注每个元素的前K个最重要的关系来提高计算效率和模型性能
# class TKSAAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(TKSAAttention, self).__init__()
#         self.num_heads = num_heads
#
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.attn_drop = nn.Dropout(0.)
#
#         self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         _, _, C, _ = q.shape
#
#         mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
#         mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
#         mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
#         mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#
#         index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
#         mask1.scatter_(-1, index, 1.)
#         attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
#
#         index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
#         mask2.scatter_(-1, index, 1.)
#         attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
#
#         index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
#         mask3.scatter_(-1, index, 1.)
#         attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))
#
#         index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
#         mask4.scatter_(-1, index, 1.)
#         attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
#
#         attn1 = attn1.softmax(dim=-1)
#         attn2 = attn2.softmax(dim=-1)
#         attn3 = attn3.softmax(dim=-1)
#         attn4 = attn4.softmax(dim=-1)
#
#         out1 = (attn1 @ v)
#         out2 = (attn2 @ v)
#         out3 = (attn3 @ v)
#         out4 = (attn4 @ v)
#
#         out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#
#
# if __name__ == '__main__':
#     block = TKSAAttention(dim=256, num_heads=2, bias=False)
#     input = torch.rand(32, 256, 224, 224)
#     output = block(input)
#     print(input.size())
#     print(output.size())

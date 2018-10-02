

class NonLocal(nn.Module):
    def __init__(self, mdim):
        super(NonLocal, self).__init__()
        self.conv_th = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_pi = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_g = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(int(mdim/2), mdim, kernel_size=1, padding=0)
 
    def forward(self, x1, x2):
        res = x1
        e1 = self.conv_th(x1)
        e1 = e1.view(-1, e1.size()[1], e1.size()[2]*e1.size()[3])
        e1 = torch.transpose(e1, 1, 2)  # b, hw1, c/2
 
        e2 = self.conv_pi(x2)
        e2 = e2.view(-1, e2.size()[1], e2.size()[2]*e2.size()[3])  # b, c/2, hw2
 
        f = torch.bmm(e1, e2) # b, hw1, hw2
        f = F.softmax(f.view(-1, f.size()[1]*f.size()[2]), dim=1).view(-1, f.size()[1], f.size()[2]) # b, hw1, hw2
 
        g2 = self.conv_g(x2) 
        g2 = g2.view(-1, g2.size()[1], g2.size()[2]*g2.size()[3])
        g2 = torch.transpose(g2, 1, 2) # b, hw2, c/2
 
        out = torch.bmm(f, g2)  # b, hw1, c/2
        out = torch.transpose(out, 1, 2).contiguous() # b, c/2, hw1
        out = out.view(-1, out.size()[1], x1.size()[2], x1.size()[3]) # b, c/2, h1, w1
        out = self.conv_out(out)  # b, c, h1, w1
 
        out = out + res
        return out
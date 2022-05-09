# Visual-Attention-Network
Implementation of Visual Attention Network (Van)  in Tensorflow 2

> A large kernel convolution can be divided into three components: a spatial local convolution (depth-wise convolution), a spatial long-range convolution (depth-wise dilation convolution), and a channel convolution (`1×1` convolution). Specifically, a `K×K` convolution is decomposed into a `⌈ K/d ⌉×⌈ K/d ⌉` depth-wise dilation convolution with dilation `d`, a `(2d − 1) × (2d − 1)` depth-wise convolution and a `1×1` convolution. Through the above decomposition, the module can capture long-range relation- ship with slight computational cost and parameters.


| Properties                | Convolution   | Self-Attention  | LKA   |
| :---:                     | :-:           | :-:             |:-:    |
| Local Receptive Field     | ◯             | ✖️              |◯      |
| Long-range Dependence     | ✖️            | ◯               |◯      |
| Spatial Adaptability      | ✖️            | ◯               |◯      |
| Channel Adaptability      | ✖️            | ✖️              |◯      |



## Usage
These codes are well-aligned with pytorch implementation.
[Pytorch code](https://github.com/Visual-Attention-Network/VAN-Classification/blob/ccdfc6883d0da136010bb8cea52bec3587ffb250/models/van.py)

```python
class LKA(layers.Layer):
  def __init__(self, dim):
    super(LKA,self).__init__()
    
    self.conv0 = layers.DepthwiseConv2D(5, padding='same',name='conv0')
    self.conv_spatial = layers.DepthwiseConv2D(7, padding='same', dilation_rate=3, name='conv_spatial')
    self.conv1 = layers.Conv2D(dim, 1, name='conv1')
    
  #@tf.function(jit_compile=True) 
  def call(self, x):
    u = x        
    attn = self.conv0(x)
    attn = self.conv_spatial(attn)
    attn = self.conv1(attn)

    return u * attn
```

```python
class Attention(layers.Layer):
  
  def __init__(self, d_model,**kwargs):
    super(Attention,self).__init__(**kwargs)

    self.proj_1 = layers.Conv2D(d_model, 1, name='proj_1')
    self.activation = tfa.layers.GELU()
    self.spatial_gating_unit = LKA(d_model)
    self.proj_2 = layers.Conv2D(d_model, 1, name='proj_2')
  
  #@tf.function(jit_compile=True) 
  def call(self, x):
    shorcut = x
    x = self.proj_1(x)
    x = self.activation(x)
    x = self.spatial_gating_unit(x)
    x = self.proj_2(x)
    x = x + shorcut
    return x
```

```python
class Block(layers.Layer):

  def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=tfa.layers.GELU):
    super(Block, self).__init__()
    
    self.norm1 = layers.BatchNormalization(name='norm1', epsilon=1.001e-5)
    self.attn = Attention(dim,name='attn')
    self.drop_path = DropPath(drop_path)

    self.norm2 = layers.BatchNormalization(name='norm2', epsilon=1.001e-5)
    
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    layer_scale_init_value = 1e-2            
    self.layer_scale_1 = self.add_weight(
      'layer_scale_1', shape=[1, 1, 1, dim], initializer=initializers.Constant(layer_scale_init_value),
      trainable=True, dtype=self.dtype)

    self.layer_scale_2 = self.add_weight(
      'layer_scale_2', shape=[1, 1, 1, dim], initializer=initializers.Constant(layer_scale_init_value),
      trainable=True, dtype=self.dtype)
  
  #@tf.function(jit_compile=True) 
  def call(self, x):
    x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
    x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
    return x
```

```python
# VAN
class VAN(layers.Layer):

  def __init__(self, img_size=224, num_classes=1000, embed_dims=[64, 128, 256, 512],
               mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer= LayerNorm,
               depths=[3, 4, 6, 3], num_stages=4, flag=False):
    
    super(VAN, self).__init__()
    if flag == False:
      self.num_classes = num_classes
    self.depths = depths
    self.num_stages = num_stages

    dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    cur = 0

    for i in range(num_stages):
      patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                      patch_size=7 if i == 0 else 3,
                                      stride=4 if i == 0 else 2,
                                      embed_dim=embed_dims[i])

      block = [Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])]
      norm = norm_layer(embed_dims[i])
      cur += depths[i]

      setattr(self, f"patch_embed{i + 1}", patch_embed)
      setattr(self, f"block{i + 1}", block)
      setattr(self, f"norm{i + 1}", norm)

    # classification head
    self.flatten = layers.GlobalAveragePooling2D(name='avg_pool')
    self.head = layers.Dense(num_classes, name='head')  

  def get_classifier(self):
    return self.head

  def reset_classifier(self, num_classes, global_pool=''):
    self.num_classes = num_classes
    self.head = layers.Dense(num_classes, name='head')  

  def forward_features(self, x):

    for i in range(self.num_stages):
      patch_embed = getattr(self, f"patch_embed{i + 1}")
      block = getattr(self, f"block{i + 1}")
      norm = getattr(self, f"norm{i + 1}")
      x = patch_embed(x)
      for blk in block:
        x = blk(x)   
      x = norm(x)

    return self.flatten(x)
  
  #@tf.function(jit_compile=True) 
  def call(self, x):
    x = self.forward_features(x)
    x = self.head(x)

    return x
```

import tensorflow.compat.v1 as tf
import tf_attention
from tensorflow.contrib import layers as tcl

color_print = "\033[7m{}\033[0m"

def dice(x, mode, name):
    with tf.variable_scope(name+"_dice"): 
        alpha = tf.get_variable(
            name="alpha",
            shape=x.shape[-1],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32
        )
        norm_x = tf.layers.batch_normalization(
            inputs=x,
            center=False,
            scale=False,
            training = mode == tf.estimator.ModeKeys.TRAIN
        )
        norm_x = tf.nn.sigmoid(norm_x)
    return norm_x * x + (1 - norm_x) * alpha * x

def mlp_layer(
    inputs,
    mode,
    mlp_cf,
    use_bn=False,
    use_ln=False,
    use_residue=False
):
    print(color_print.format(f"mlp_layer - inputs:{inputs}"))
    mlp_inp = inputs
    for i, (unit, act, l2_reg) in enumerate(mlp_cf):
        mlp_oup = tf.layers.Dense(
            unit,
            activation=None,
            kernel_regularizer=tcl.l2_regularizer(l2_reg),
            kernel_initializer=tf.glorot_uniform_initializer
        )(mlp_inp)
        if use_bn:
            mlp_oup = tf.layers.batch_normalization(
                mlp_oup,
                is_training=mode==tf.estimator.ModeKeys.TRAIN
            )
        elif use_ln:
            mlp_oup /= (tf.math.reduce_std(mlp_oup, axis=-1, keepdims=True)+1e-9)
        mlp_oup = act(mlp_oup, mode)
        mlp_inp = mlp_oup if not use_residue else mlp_inp + mlp_oup
        tf.summary.histogram(f"MLP-{i}", mlp_inp)
        print(f'- layer_{i}:{mlp_inp}')

    return tf.layers.Dense(1)(mlp_inp)

def mmoe_layer(
    expert_inputs,
    mode,
    expert_cf,
    task_cf
):
    def mlp_op(inp, mlp_cf, prefix):
        oup = inp
        for i, (unit, act, l2_reg) in enumerate(mlp_cf):
            oup = tf.layers.Dense(
                unit,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(oup)
            tf.summary.histogram(f"{prefix}_MLP_{i}", oup)

        return oup
            
    expert_inp = tf.concat(list(expert_inputs.values()), axis=-1)
    expert_oup = tf.stack([mlp_op(expert_inp, expert_cf["mlp_cf"], f"MMOE_expert_{i}") for i in range(expert_cf['expert_num'])], axis=1)
    print(color_print.format(f"mmoe_layer - expert_inp:{expert_inp}"))
    print(f"- expert_oup:{expert_oup}")

    mmoe_oup = {}
    for task_name, mlp_cf in task_cf.items():
        gate_oup = expert_inp
        for i, (unit, act, l2_reg) in enumerate(mlp_cf):
            gate_oup = tf.layers.Dense(
                units=unit,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(gate_oup)
            tf.summary.histogram(f"MMOE_{task_name}_gate_oup_{i}", gate_oup)
        gate_oup = tf.layers.Dense(expert_cf['expert_num'], activation=tf.nn.sigmoid)(gate_oup)
        task_oup = tf.einsum("bcd,bc->bd", expert_oup, gate_oup)

        print(f'- task_{task_name}_oup:{task_oup}')
        tf.summary.histogram(f"MMOE_{task_name}_task_oup", task_oup)
        mmoe_oup[task_name] = task_oup

    return mmoe_oup

def ple_layer(
    inps, 
    mode, 
    layer_num,
    tasks, 
    share_dim,
    task_dim, 
    expert_num, 
    expert_dim,
    l2_reg,
):
    def cgc_unit(inp, prefix, is_last=False):
        task_expert = {}
        for task in tasks:
            task_expert[task] = tf.stack([tf.layers.Dense(
                units=expert_dim,
                activation=lambda x : dice(x, mode, f"{prefix}_{task}_expert_{i}"),
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(inp[task]) for i in range(expert_num)], axis=1) 
            tf.summary.histogram(f"{prefix}_expert_{task}",task_expert[task]) 
            print(f'- {prefix}_expert_{task}:{task_expert[task]}')

        share_expert = tf.stack([tf.layers.Dense(
            units=expert_dim,
            activation=lambda x : dice(x, mode, f"{prefix}_share_expert_{i}"),
            kernel_regularizer=tcl.l2_regularizer(l2_reg),
            kernel_initializer=tf.glorot_uniform_initializer
        )(inp["share"]) for i in range(expert_num)], axis=1)
        tf.summary.histogram(f"{prefix}_expert_share", share_expert)
        print(f'- {prefix}_expert_share:{share_expert}')
    
        task_oup = {}
        for task in tasks:
            """
            gate_oup = tf.layers.Dense(
                units=task_dim,
                activation=tf.nn.leaky_relu,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(inp[task])
            """
            gate_logit = tf.layers.Dense(
                units=expert_num * 2,
                activation=tf.nn.sigmoid,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(inp[task])
            task_oup[task] = tf.einsum("bc,bcd->bd", gate_logit,
                tf.concat([share_expert, task_expert[task]], axis=1))

            #tf.summary.histogram(f"{prefix}_gate_{task}_oup", gate_oup)
            tf.summary.histogram(f"{prefix}_gate_{task}_logit", gate_logit)
            tf.summary.histogram(f"{prefix}_task_{task}_oup", task_oup[task])
            #print(f'- {prefix}_gate_oup_{task}:{gate_oup}')
            print(f'- {prefix}_gate_logit_{task}:{gate_logit}')
            print(f'- {prefix}_task_oup_{task}:{task_oup[task]}')

        if not is_last:
            """
            gate_oup = tf.layers.Dense(
                units=task_dim,
                activation=tf.nn.leaky_relu,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kerlnel_initializer=tf.glorot_uniform_initializer
            )(inp["share"])
            """
            gate_logit = tf.layers.Dense(
                units=(len(tasks)+1) * expert_num,
                activation=tf.nn.sigmoid,
                kernel_regularizer=tcl.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_uniform_initializer
            )(inp["share"])
            taks_oup["share"] = tf.einsum("bc,bcd->bcd", gate_logit,
                tf.concat([share_expert]+list(task_expert.values()), axis=-1))
            
            #tf.summary.histogram(f"{prefix}_gate_share_oup", gate_oup)
            tf.summary.histogram(f"{prefix}_gate_share_logit", gate_logit)
            tf.summary.histogram(f"{prefix}_task_share_oup", task_oup["share"])
            #print(f'- {prefix}_gate_oup_share:{gate_oup}')
            print(f'- {prefix}_gate_logit_share:{gate_logit}')
            print(f"- {prefix}_task_oup_share:{task_oup['share']}")

        
        return task_oup
    
    print(color_print.format(f"ple_layer - inputs:{inps}"))
    share_inp = tf.layers.dense(
        inps, share_dim, activation=lambda x : dice(x, mode, "ple_share_dice"))
    layer_oup = {task:share_inp for task in tasks + ["share"]}
    for i in range(layer_num):
        layer_oup = cgc_unit(layer_oup, f"cgc_{i}", i == layer_num - 1)
        print(f'- cgc_{i}:{layer_oup}')

    return layer_oup

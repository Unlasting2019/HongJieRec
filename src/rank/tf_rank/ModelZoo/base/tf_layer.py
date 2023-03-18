import tensorflow.compat.v1 as tf
import tf_attention
from tensorflow.contrib import layers as tcl

color_print = "\033[7m{}\033[0m"

def pc_vec(pc_vec, out_dim, pc_dim=100):
    return tf.layers.dense(
        tf.layers.dense(
            pc_vec,
            pc_dim,
            activation=tf.nn.leaky_relu
        ), out_dim, activation=tf.nn.sigmoid
    ) * 2 

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

class LayerOp:
    def __init__(self, features, mode, pc_vec):
        self.features = features
        self.mode = mode
        self.pc_vec = pc_vec

    def mlp_layer(self, inps, mlp_cf, pc=False):
        print(color_print.format(f"mlp_layer - inputs:{inps}"))
        if pc:
            inp_pc = pc_vec(self.pc_vec, inps.shape[-1])
            mlp_oup = inps * inp_pc
            tf.summary.histogram("mlp_layer_inp_pc", inp_pc)
            tf.summary.histogram("mlp_layer_inp", mlp_oup)
        else:
            mlp_oup = inps

        for i, (unit, act) in enumerate(mlp_cf):
            mlp_oup1 = tf.layers.dense(mlp_oup, unit)
            mlp_oup2 = mlp_oup1 / tf.math.reduce_std(mlp_oup, -1, True)+1e-9
            mlp_oup3 = act(mlp_oup2, self.mode)
            if pc:
                oup_pc = pc_vec(self.pc_vec if pc else None, unit)
                mlp_oup = mlp_oup3 * oup_pc
                tf.summary.histogram(f"mlp_layer{i}_oup_pc", oup_pc)
            else:
                mlp_oup = mlp_oup3
            tf.summary.histogram(f"mlp_layer{i}_oup1", mlp_oup1)
            tf.summary.histogram(f"mlp_layer{i}_oup2", mlp_oup2)
            tf.summary.histogram(f"mlp_layer{i}_oup3", mlp_oup3)
            tf.summary.histogram(f"mlp_layer{i}_oup", mlp_oup)
            print(f'- layer_{i}:{mlp_oup}')

        return mlp_oup

    def ple_layer(self,inps,pc,layer_num,tasks,expert_num,expert_dim,task_dim):
        def cgc_unit(inp, prefix, is_last=False):
            task_expert = {}
            for task in tasks:
                experts = tf.reshape(tf.layers.dense(inp[task], expert_dim*expert_num), [-1, expert_num, expert_dim])
                experts /= (tf.math.reduce_std(experts, -1, True)+1e-9)
                experts = dice(experts, self.mode, f"{prefix}_{task}_expert_dice")
                if pc:
                    expert_pc = tf.reshape(pc_vec(self.pc_vec, expert_num * expert_dim, 100), [-1, expert_num, expert_dim])
                    experts *= expert_pc
                    tf.summary.histogram(f"{prefix}_expert_{task}_pc", expert_pc)

                task_expert[task] = experts
                tf.summary.histogram(f"{prefix}_expert_{task}",task_expert[task]) 
                print(f'- {prefix}_expert_{task}:{task_expert[task]}')

            share_expert = tf.reshape(tf.layers.dense(inp["share"], expert_dim), [-1, expert_num, expert_dim])
            share_expert /= (tf.math.reduce_std(share_expert, -1, True)+1e-9)
            share_expert = dice(share_expert, self.mode, f"{prefix}_share_expert_dice")
            if pc:
                share_pc = tf.reshape(pc_vec(self.pc_vec, expert_num * expert_dim, 100), [-1, expert_num, expert_dim])
                share_expert *= share_pc
                tf.summary.histogram(f"{prefix}_expert_share_pc", share_pc)

            tf.summary.histogram(f"{prefix}_expert_share", share_expert)
            print(f'- {prefix}_expert_share:{share_expert}')
        
            task_oup = {}
            for task in tasks:
                gate_logit = tf.layers.dense(inp[task], task_dim, activation=tf.nn.leaky_relu)
                gate_prob  = tf.layers.dense(gate_logit, expert_num*2, activation=tf.nn.sigmoid)
                task_oup[task] = tf.einsum("bc,bcd->bd", gate_prob,
                    tf.concat([share_expert, task_expert[task]], axis=1))
                
                for i in range(expert_num*2):
                    tf.summary.scalar(f"{prefix}_gate_{task}_prob", tf.reduce_mean(gate_prob[:, i]))
                tf.summary.histogram(f"{prefix}_task_{task}_logit", gate_logit)
                tf.summary.histogram(f"{prefix}_task_{task}_oup", task_oup[task])
                print(f'- {prefix}_gate_logit_{task}:{gate_logit}')
                print(f'- {prefix}_task_oup_{task}:{task_oup[task]}')

            if not is_last:
                gate_logit = tf.layers.dense(inp["share"], task_dim,activation=tf.nn.leaky_relu)
                gate_prob = tf.layers.dense(gate_logit, (len(tasks)+1)*expert_num, activation=tf.nn.sigmoid)
                taks_oup["share"] = tf.einsum("bc,bcd->bcd", gate_prob,
                    tf.concat([share_expert]+list(task_expert.values()), axis=-1))

                tf.summary.histogram(f"{prefix}_task_share_logit", gate_logit)
                tf.summary.histogram(f"{prefix}_task_share_prob", gate_prob)
                tf.summary.histogram(f"{prefix}_task_share_oup", task_oup["share"])
                print(f'- {prefix}_gate_logit_share:{gate_logit}')
                print(f"- {prefix}_task_oup_share:{task_oup['share']}")

            
            return task_oup

        print(color_print.format(f"ple_layer - inputs:{inps}"))
        layer_oup = {task:inps for task in tasks + ["share"]}
        for i in range(layer_num):
            layer_oup = cgc_unit(layer_oup, f"cgc_{i}", i == layer_num - 1)
            print(f'- cgc_{i}:{layer_oup}')

        return layer_oup

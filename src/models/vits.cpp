#include "models.h"

llm_build_vits::llm_build_vits(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = nullptr;
    if(ubatch.token){
        inp_pos = build_inp_pos();
        ggml_tensor* pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        ggml_tensor *scaled_token_emb = ggml_scale(ctx0, inpL, hparams.x_scale);
        ggml_tensor *scaled_pos_emb = ggml_scale(ctx0, pos, hparams.alpha);
        inpL = ggml_add(ctx0, scaled_token_emb, scaled_pos_emb);
    }

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = inpL;

        // self-attention
        {
            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            if (model.layers[il].bqkv){
                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);
            }

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 2*sizeof(float)*(n_embd));
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);

        // feed-forward network
        ffn_inp = build_norm(ffn_inp,
                model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                LLM_NORM, il);
        cb(ffn_inp, "ffn_norm", il);

        cur = build_ffn(ffn_inp,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_RELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM, il);

        cb(cur, "l_out", il);

        inpL = cur;
    }

    // lm_head
    cur = build_lora_mm(model.output, cur);
    if (model.output_b) {
        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_output", -1);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

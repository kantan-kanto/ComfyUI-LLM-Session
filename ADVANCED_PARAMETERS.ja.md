# 高度なパラメータ

このページでは、Simple ノード向けの JSON による高度なパラメーターの設定について説明します。

これらの設定は、Simple ノードの `config_path` で選択した JSON ファイルから使用します。


## サポートされている高度な生成設定

高度なパラメーターの候補は以下のサンプルに列挙しています。

- `config/simple_advanced.example.json`

このうり現在サポートしているのは`advanced_generation_kwargs`セクションの `seed`と"advanced_summary_generation_kwargs"の`seed` のみです。

```json
{
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}

###　`seed` を一般パラメーターにせず、高度なパラメーターに舌理由

`seed` は llama-cpp-python の生成処理に渡されます。これは確率的サンプリングの再現性を高めますが、全体的な決定論を保証するものではありません。

同じモデル、プロンプト、画像入力、生成設定、セッション状態、runtime cache の挙動、backend の挙動が揃っている場合、固定 seed によって `temperature` が `0` より大きい場合でも出力を再現できることがあります。

### 'Seed'以外の高度なパラメーター

`config/simple_advanced.example.json`に列挙した'seed'以外の未サポートのパラメーターは現状では無視されます。`log_level` が `minimal` ではない場合、ノードはそれらの未サポートキーについて warning を出力します。


## "advanced_summary_generation_kwargs"について

Summary 生成には専用の高度なセクションがあります。

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

summary 用のパラメータは 以下のように通常生成とは別に定められています。

summary temperature: 0.2
summary max_tokens: max_tokens_summary、デフォルト 128
summary top_p: llama.cpp 側のデフォルト（ノード側では未指定）
summary repeat_penalty: llama.cpp 側のデフォルト（ノード側では未指定）

"advanced_summary_generation_kwargs"ではこれらのパラメーターを、オーバーライド可能にする予定です。


## 再現性に関する注意

再現性テストを行う場合は、まず次の設定を試してください。
私の環境では`LlamaTrieCache` が設定してあると、同じ seed、プロンプト、モデル、画像を使用していても出力が変わることがありました。再現性を確認したい場合は、`runtime_cache: "off"` を推奨します。

```json
{
  "runtime_cache": "off",
  "reset_session": true,
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

## 固定 seed でも出力が変わる場合

固定 seed が制御するのは、サンプリングの乱数源だけです。実効的な生成入力は一致している必要があります。

まず次を確認してください。

- `runtime_cache` を `"off"` に設定してください。
- `reset_session: true` または新しい `session_id` を使用してください。
- 同じモデルファイル、mmproj ファイル、画像入力、プロンプト、config を使用してください。
- 履歴や summary テキストがプロンプトを変えていないことを確認してください。
- summary の再現性が重要な場合は、`advanced_summary_generation_kwargs.seed` を設定してください。
- 保存された履歴の `params` を比較し、実際に有効になった設定を確認してください。
- backend、ハードウェア、llama-cpp-python の違いによっても、出力が変わることがあります。

## 履歴への記録

適用された高度な生成設定は、保存される各 turn の `params` に記録されます。

例:

```json
{
  "params": {
    "advanced_generation_kwargs": {
      "seed": 12345
    }
  }
}
```

summary の高度な設定が適用された場合、それらも記録されます。

```json
{
  "params": {
    "advanced_summary_generation_kwargs": {
      "seed": 456
    }
  }
}
```

## まだ有効ではない項目

`config/simple_advanced.example.json` には、将来の高度な backend 設定や生成設定のための実験的なフィールドを含んでいます。現時点でサポートされている高度な生成キーは `seed` のみです。
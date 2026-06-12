# 高度なパラメータ

このページでは、Simple ノード向けの高度な JSON 設定について説明します。

これらの設定は、Simple ノードの `config_path` で選択した JSON ファイルから使用します。
より完全な例については、次を参照してください。

- `config/simple_advanced.example.json`

## サポートされている高度な生成設定

高度な生成セクションで現在サポートしているのは `seed` のみです。

```json
{
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

`seed` は llama-cpp-python の生成処理に渡されます。これは確率的サンプリングの再現性を高めますが、全体的な決定論を保証するものではありません。

同じモデル、プロンプト、画像入力、生成設定、セッション状態、runtime cache の挙動、backend の挙動が揃っている場合、固定 seed によって `temperature` が `0` より大きい場合でも出力を再現できることがあります。

`advanced_generation_kwargs` 内の未サポートのキーは無視されます。`log_level` が `minimal` ではない場合、ノードはそれらの未サポートキーについて warning を出力します。

## Summary 生成設定

Summary 生成には専用の高度なセクションがあります。

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

通常生成の seed は summary に自動的には再利用されません。summary の再現性が必要な場合は、`advanced_summary_generation_kwargs.seed` を明示的に設定してください。

`advanced_summary_generation_kwargs` 内の未サポートのキーは無視されます。`log_level` が `minimal` ではない場合、ノードはそれらの未サポートキーについて warning を出力します。

## 再現性に関する注意

厳密な再現性テストを行う場合は、次を使用してください。

```json
{
  "runtime_cache": "off",
  "reset_session": true,
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

`LlamaTrieCache` などの runtime cache は、同じ seed、プロンプト、モデル、画像を使用していても出力を変えることがあります。正確な再現性を確認したい場合は、`runtime_cache: "off"` を推奨します。

ログメッセージ `Using cached model` は、すでにロード済みのモデルインスタンスが再利用されたことだけを意味します。これは runtime prompt/KV cache の挙動とは別のものです。正確な再現性は、`runtime_cache`、セッション状態、プロンプト/履歴の内容、モデル、mmproj、画像入力、生成設定により直接的に依存します。

再現性を確認する場合:

- 同じモデルと mmproj を使用してください。
- 同じプロンプト、画像入力、生成設定を使用してください。
- `runtime_cache: "off"` を使用してください。
- `reset_session: true` または新しい session id を使用してください。
- 保存された履歴の `params` に、期待する `advanced_generation_kwargs.seed` が含まれていることを確認してください。

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

`config/simple_advanced.example.json` には、将来の高度な backend 設定や生成設定のための実験的なフィールドが含まれている場合があります。現時点でサポートされている高度な生成キーは `seed` のみです。

特に、広範な backend kwargs や未サポートの generation kwargs は自動的には転送されません。それらは、各オプションについて llama-cpp-python の挙動に照らして検証した後にのみ有効化すべきです。

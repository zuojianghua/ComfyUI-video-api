import { app } from "../../../scripts/app.js";

// Kling 节点：切换模型时把「时长 / 清晰度」下拉收敛到该模型支持的值（如 kling-2.6 仅 5/10 秒、无 4k）。
// 规则唯一来源是 Python 端 kling_api.MODEL_RULES，经 model_name 输入的 kling_rules 下发，这里不另写一份。
const PREFERRED = { duration: "5", resolution: "1080p" };

app.registerExtension({
  name: "ComfyUI-video-api.KlingOptions",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KlingImageToVideo") return;
    const rules = nodeData.input?.required?.model_name?.[1]?.kling_rules;
    if (!rules) return;

    function applyRules(node) {
      const widget = (name) => node.widgets?.find((w) => w.name === name);
      const model = widget("model_name");
      const rule = model && rules[model.value];
      if (!rule) return;
      for (const [name, values] of [["duration", rule.durations], ["resolution", rule.resolutions]]) {
        const w = widget(name);
        if (!w) continue;
        w.options.values = values;
        if (!values.includes(w.value)) {
          w.value = values.includes(PREFERRED[name]) ? PREFERRED[name] : values[0];
        }
      }
      node.setDirtyCanvas(true, true);
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const model = this.widgets?.find((w) => w.name === "model_name");
      if (model) {
        const callback = model.callback;
        const node = this;
        model.callback = function () {
          const r = callback?.apply(this, arguments);
          applyRules(node);
          return r;
        };
      }
      applyRules(this);
      return result;
    };

    // 打开已保存的工作流时，控件值在 onNodeCreated 之后才恢复，需再收敛一次
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      applyRules(this);
      return result;
    };
  },
});

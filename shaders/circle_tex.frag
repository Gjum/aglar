uniform sampler2D tex0;
varying vec2 texCoord0;

void main() {
    // gl_Color.a == 0.0 for textured circles
    vec4 colorOuter = gl_Color;
    colorOuter.a = 1.0;
    vec4 colorInner = (1.-gl_Color.a) * vec4(texture2D(tex0, texCoord0).rgb, 1.0);
    colorInner += (1.-colorInner.a) * colorOuter;
    float len = length(texCoord0 - vec2(0.5, 0.5));
    float circleStepInner = 1.0 - step(0.47, len);
    float circleStepOuter = 1.0 - step(0.5, len) - circleStepInner;
    gl_FragColor = circleStepInner * colorInner
                 + circleStepOuter * colorOuter;
    gl_FragColor *= .6;
}

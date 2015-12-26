varying vec2 texCoord0;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_FrontColor = gl_Color;
    texCoord0 = gl_MultiTexCoord0.xy;
}

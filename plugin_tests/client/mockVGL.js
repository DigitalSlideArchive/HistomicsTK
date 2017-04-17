/* eslint-disable camelcase */
/* eslint-disable underscore/prefer-constant */

window.mockVGLRenderer = function mockVGLRenderer(supported) {
    var vgl = window.vgl;

    if (supported === undefined) {
        supported = true;
    }

    if (vgl._mocked) {
        throw new Error('VGL renderer already mocked');
    }

    var mockCounts = {};
    var count = function (name) {
        mockCounts[name] = (mockCounts[name] || 0) + 1;
    };
    var noop = function (name) {
        return function () {
            count(name);
        };
    };
    var _id = 0,
        incID = function (name) {
            return function () {
                count(name);
                _id += 1;
                return _id;
            };
        };
    /* The context largely does nothing. */
    var m_context = {
        activeTexture: noop('activeTexture'),
        attachShader: noop('attachShader'),
        bindAttribLocation: noop('bindAttribLocation'),
        bindBuffer: noop('bindBuffer'),
        bindFramebuffer: noop('bindFramebuffer'),
        bindTexture: noop('bindTexture'),
        blendFuncSeparate: noop('blendFuncSeparate'),
        bufferData: noop('bufferData'),
        bufferSubData: noop('bufferSubData'),
        checkFramebufferStatus: function (key) {
            count('checkFramebufferStatus');
            if (key === vgl.GL.FRAMEBUFFER) {
                return vgl.GL.FRAMEBUFFER_COMPLETE;
            }
        },
        clear: noop('clear'),
        clearColor: noop('clearColor'),
        clearDepth: noop('clearDepth'),
        compileShader: noop('compileShader'),
        createBuffer: incID('createBuffer'),
        createFramebuffer: noop('createFramebuffer'),
        createProgram: incID('createProgram'),
        createShader: incID('createShader'),
        createTexture: incID('createTexture'),
        deleteBuffer: noop('deleteBuffer'),
        deleteProgram: noop('deleteProgram'),
        deleteShader: noop('deleteShader'),
        deleteTexture: noop('deleteTexture'),
        depthFunc: noop('depthFunc'),
        disable: noop('disable'),
        disableVertexAttribArray: noop('disableVertexAttribArray'),
        drawArrays: noop('drawArrays'),
        enable: noop('enable'),
        enableVertexAttribArray: noop('enableVertexAttribArray'),
        finish: noop('finish'),
        getExtension: incID('getExtension'),
        getParameter: function (key) {
            count('getParameter');
            if (key === vgl.GL.DEPTH_BITS) {
                return 16;
            }
        },
        getProgramParameter: function (id, key) {
            count('getProgramParameter');
            if (key === vgl.GL.LINK_STATUS) {
                return true;
            }
        },
        getShaderInfoLog: function () {
            count('getShaderInfoLog');
            return 'log';
        },
        getShaderParameter: function (id, key) {
            count('getShaderParameter');
            if (key === vgl.GL.COMPILE_STATUS) {
                return true;
            }
        },
        getUniformLocation: incID('getUniformLocation'),
        isEnabled: function (key) {
            count('isEnabled');
            if (key === vgl.GL.BLEND) {
                return true;
            }
        },
        linkProgram: noop('linkProgram'),
        pixelStorei: noop('pixelStorei'),
        shaderSource: noop('shaderSource'),
        texImage2D: noop('texImage2D'),
        texParameteri: noop('texParameteri'),
        uniform1iv: noop('uniform1iv'),
        uniform1fv: noop('uniform1fv'),
        uniform2fv: noop('uniform2fv'),
        uniform3fv: noop('uniform3fv'),
        uniform4fv: noop('uniform4fv'),
        uniformMatrix3fv: noop('uniformMatrix3fv'),
        uniformMatrix4fv: noop('uniformMatrix4fv'),
        useProgram: noop('useProgram'),
        vertexAttribPointer: noop('vertexAttribPointer'),
        vertexAttrib3fv: noop('vertexAttrib3fv'),
        viewport: noop('viewport')
    };

    /* Our mock has only a single renderWindow */
    var m_renderWindow = vgl.renderWindow();
    m_renderWindow._setup = function () {
        return true;
    };
    m_renderWindow.context = function () {
        return m_context;
    };
    vgl.renderWindow = function () {
        return m_renderWindow;
    };
    window.geo.gl.vglRenderer.supported = function () {
        return !!supported;
    };

    vgl._mocked = true;
    vgl.mockCounts = function () {
        return mockCounts;
    };
};

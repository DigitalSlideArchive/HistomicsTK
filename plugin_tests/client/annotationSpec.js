/* eslint-disable camelcase */
/* eslint-disable underscore/prefer-constant */
girderTest.importStylesheet(
    '/static/built/plugins/jobs/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/slicer_cli_web/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/large_image/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/HistomicsTK/plugin.min.css'
);
girderTest.addCoveredScripts([
    '/clients/web/static/built/plugins/HistomicsTK/plugin.min.js'
]);

var app;
var geojsMap;

function mockVGLRenderer(supported) {
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
}

girderTest.promise.then(function () {
    $('body').css('overflow', 'hidden');
    girder.router.enabled(false);
    girder.events.trigger('g:appload.before');
    app = new girder.plugins.HistomicsTK.App({
        el: 'body',
        parentView: null
    });
    app.bindRoutes();
    girder.events.trigger('g:appload.after');
});

$(function () {
    describe('Annotation tests', function () {
        describe('setup', function () {
            it('login', function () {
                girderTest.waitForLoad();

                runs(function () {
                    $('.g-login').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    $('#g-login').val('admin');
                    $('#g-password').val('password');
                    $('#g-login-button').click();
                });

                waitsFor(function () {
                    return $('.h-user-dropdown-link').length > 0;
                }, 'user to be logged in');
            });

            it('open image', function () {
                $('.h-open-image').click();
                girderTest.waitForDialog();

                runs(function () {
                    $('#g-root-selector').val(
                        girder.auth.getCurrentUser().id
                    ).trigger('change');
                });

                waitsFor(function () {
                    return $('#g-dialog-container .g-folder-list-link').length > 0;
                }, 'Hierarchy widget to render');

                runs(function () {
                    $('.g-folder-list-link:contains("Public")').click();
                });

                waitsFor(function () {
                    return $('.g-item-list-link').length > 0;
                }, 'item list to load');

                runs(function () {
                    $('.g-item-list-link').click();
                    $('.g-submit-button').click();
                });

                waitsFor(function () {
                    return $('.geojs-layer.active').length > 0;
                }, 'image to load');

                runs(function () {
                    geojsMap = app.bodyView.viewer;
                    mockVGLRenderer(true);
                });
            });
        });

        describe('Annotation panel', function () {
            it('panel is rendered', function () {
                expect($('.h-annotation-selector .s-panel-title').text()).toMatch(/Annotations/);
                expect($('.h-annotation-selector .h-annotation').length).toBe(0);
            });
        });

        describe('Draw panel', function () {
            it('draw a point', function () {
                runs(function () {
                    $('.h-draw[data-type="point"]').click();
                });

                waitsFor(function () {
                    return $('.geojs-map.annotation-input').length > 0;
                }, 'draw mode to activate');
                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousedown', {
                        map: {x: 100, y: 100},
                        button: 'left'
                    });
                    interactor.simulateEvent('mouseup', {
                        map: {x: 100, y: 100},
                        button: 'left'
                    });
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element').length === 1;
                }, 'point to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element .h-element-label').text()).toBe('point');
                });
            });

            it('edit a point element', function () {
                runs(function () {
                    $('.h-elements-container .h-edit-element').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    expect($('#g-dialog-container .modal-title').text()).toBe('Edit annotation');
                    $('#g-dialog-container #h-element-label').val('test');
                    $('#g-dialog-container .h-submit').click();
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element .h-element-label').text() === 'test';
                }, 'label to change');
            });

            it('draw another point', function () {
                runs(function () {
                    $('.h-draw[data-type="point"]').click();
                });

                waitsFor(function () {
                    return $('.geojs-map.annotation-input').length > 0;
                }, 'draw mode to activate');
                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousedown', {
                        map: {x: 200, y: 200},
                        button: 'left'
                    });
                    interactor.simulateEvent('mouseup', {
                        map: {x: 200, y: 200},
                        button: 'left'
                    });
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element').length === 2;
                }, 'rectangle to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element:last .h-element-label').text()).toBe('point');
                });
            });

            it('delete the second point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);
            });

            it('save the point annotation', function () {
                runs(function () {
                    $('.h-draw-widget .h-save-annotation').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    $('#g-dialog-container #h-annotation-name').val('single point');
                    $('#g-dialog-container .h-submit').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation-name').text()).toBe('single point');
                    expect($('.h-draw-widget .h-save-widget').length).toBe(0);
                });
            });
        });
    });
});

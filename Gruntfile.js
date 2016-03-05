/**
 * Copyright 2015 Kitware Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
module.exports = function (grunt) {
    grunt.config.merge({
        histomics: {
            root: '<%= pluginDir %>/HistomicsTK',
            extjs: '<%= histomics.root %>/web_client/js/ext',
            extcss: '<%= histomics.root %>/web_client/stylesheets/ext',
            extextra: '<%= histomics.root %>/web_client/extra',
            npm: '<%= histomics.root %>/node_modules',
            external: '<%= histomics.root %>/web_main',
            static: '<%= staticDir %>/built/plugins/HistomicsTK'
        },
        uglify: {
            'histomics-main': {
                files: [
                    {
                        src: [
                            '<%= histomics.external %>/main.js'
                        ],
                        dest: '<%= histomics.static %>/histomics.main.min.js'
                    }
                ]
            }
        },
        jade: {
            'plugin-HistomicsTK': {
                options: {
                    namespace: 'histomicstk.templates'
                }
            }
        },
        copy: {
            'bootstrap-slider': {
                files: [{
                    '<%= histomics.extjs %>/bootstrap-slider.js': '<%= histomics.npm %>/bootstrap-slider/dist/bootstrap-slider.js',
                    '<%= histomics.extcss %>/bootstrap-slider.css': '<%= histomics.npm %>/bootstrap-slider/dist/css/bootstrap-slider.css'
                }]
            },
            'bootstrap-colorpicker': {
                files: [{
                    '<%= histomics.extjs %>/bootstrap-colorpicker.js': '<%= histomics.npm %>/bootstrap-colorpicker/dist/js/bootstrap-colorpicker.js',
                    '<%= histomics.extcss %>/bootstrap-colorpicker.css': '<%= histomics.npm %>/bootstrap-colorpicker/dist/css/bootstrap-colorpicker.css'
                }, {
                    expand: true,
                    cwd: '<%= histomics.npm %>/bootstrap-colorpicker/dist/img',
                    src: ['bootstrap-colorpicker/*.png'],
                    dest: '<%= histomics.extextra %>'
                }]
            },
            'backbone.localStorage': {
                files: [{
                    '<%= histomics.extjs %>/backbone.localStorage.js': '<%= histomics.npm %>/backbone.localstorage/backbone.localStorage.js'
                }]
            }
        },
        stylus: {
            'plugin-HistomicsTK': {
                options: {
                    'include css': true
                }
            }
        },
        init: {
            'copy:bootstrap-slider': {
                dependencies: [
                    'shell:plugin-HistomicsTK'
                ]
            },
            'copy:bootstrap-colorpicker': {
                dependencies: [
                    'shell:plugin-HistomicsTK'
                ]
            },
            'copy:backbone.localStorage': {
                dependencies: [
                    'shell:plugin-HistomicsTK'
                ]
            }
        },
        default: {
            'uglify:histomics-main': {}
        },
        watch: {
            'plugin-histomics-uglify-main': {
                files: ['<%= histomics.external %>/**/*.js'],
                tasks: ['uglify:histomics-main']
            }
        }
    });
};

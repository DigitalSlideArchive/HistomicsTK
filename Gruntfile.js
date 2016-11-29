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
            'bootstrap-submenu': {
                files: [{
                    '<%= histomics.extjs %>/bootstrap-submenu.js': '<%= histomics.npm %>/bootstrap-submenu/dist/js/bootstrap-submenu.js',
                    '<%= histomics.extcss %>/bootstrap-submenu.css': '<%= histomics.npm %>/bootstrap-submenu/dist/css/bootstrap-submenu.css'
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
            'copy:bootstrap-submenu': {
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

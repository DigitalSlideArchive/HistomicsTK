import View from 'girder/views/View';

import PluginConfigBreadcrumbWidget from 'girder/views/widgets/PluginConfigBreadcrumbWidget';
import { restRequest } from 'girder/rest';
import events from 'girder/events';
import router from 'girder/router';

import ConfigViewTemplate from '../../templates/body/configView.pug';
import '../../stylesheets/body/configView.styl';

/**
 * Show the default quota settings for users and collections.
 */
var ConfigView = View.extend({
    events: {
        'click #g-histomicstk-save': function (event) {
            this.$('#g-histomicstk-error-message').text('');
            this._saveSettings([{
                key: 'histomicstk.default_draw_styles',
                value: this.$('#g-histomicstk-default-draw-styles').val()
            }]);
        },
        'click #g-histomicstk-cancel': function (event) {
            router.navigate('plugins', {trigger: true});
        }
    },
    initialize: function () {
        this.breadcrumb = new PluginConfigBreadcrumbWidget({
            pluginName: 'HistomicsTK',
            parentView: this
        });

        restRequest({
            method: 'GET',
            url: 'system/setting',
            data: {
                list: JSON.stringify([
                    'histomicstk.default_draw_styles'
                ])
            }
        }).done((resp) => {
            this.settings = resp;
            this.render();
        });
    },

    render: function () {
        this.$el.html(ConfigViewTemplate({
            settings: this.settings
        }));
        this.breadcrumb.setElement(this.$('.g-config-breadcrumb-container')).render();
        return this;
    },

    _saveSettings: function (settings) {
        return restRequest({
            method: 'PUT',
            url: 'system/setting',
            data: {
                list: JSON.stringify(settings)
            },
            error: null
        }).done(() => {
            events.trigger('g:alert', {
                icon: 'ok',
                text: 'Settings saved.',
                type: 'success',
                timeout: 4000
            });
        }).fail((resp) => {
            this.$('#g-histomicstk-error-message').text(
                resp.responseJSON.message
            );
        });
    }
});

export default ConfigView;

import _ from 'underscore';
import { restRequest } from 'girder/rest';

import events from '../../events';
import router from '../../router';
import View from '../View';
import headerAnalysesTemplate from '../../templates/layout/headerAnalyses.pug';
import '../../stylesheets/layout/headerAnalyses.styl';

import 'bootstrap-submenu/dist/js/bootstrap-submenu';
import 'bootstrap-submenu/dist/css/bootstrap-submenu.css';

var HeaderUserView = View.extend({
    events: {
        'click .h-analysis-item': '_setAnalysis'
    },
    initialize() {
        this.image = null;
        this.listenTo(events, 'h:imageOpened', function (image) {
            this.image = image;
            this.render();
        });
    },
    render() {
        if (this.image) {
            restRequest({
                url: 'HistomicsTK/HistomicsTK/docker_image'
            }).then((analyses) => {
                if (_.keys(analyses || {}).length > 0) {
                    this.$el.removeClass('hidden');
                    this.$el.html(headerAnalysesTemplate({
                        analyses: analyses || {}
                    }));
                    this.$('.h-analyses-dropdown-link').submenupicker();
                } else {
                    this.$el.addClass('hidden');
                }
                return null;
            });
        } else {
            this.$el.addClass('hidden');
        }
        return this;
    },
    _setAnalysis(evt) {
        evt.preventDefault();
        var target = $(evt.currentTarget).data();

        router.setQuery('analysis', target.api, {trigger: true});
    }
});

export default HeaderUserView;

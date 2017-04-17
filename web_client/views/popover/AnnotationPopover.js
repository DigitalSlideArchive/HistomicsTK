import _ from 'underscore';

import View from '../View';
import { restRequest } from 'girder/rest';

import ElementCollection from 'girder_plugins/large_image/collections/ElementCollection';
import annotationPopover from '../../templates/popover/annotationPopover.pug';
import '../../stylesheets/popover/annotationPopover.styl';

var AnnotationPopover = View.extend({
    initialize(settings) {
        if (settings.debounce) {
            this.position = _.debounce(this.position, settings.debounce);
        }

        $('body').on('mousemove', '.h-image-view-body', (evt) => this.position(evt));
        $('body').on('mouseout', '.h-image-view-body', () => this._hide());
        $('body').on('mouseover', '.h-image-view-body', () => this._show());

        this._hidden = !settings.visible;
        this._users = {};
        this.collection = new ElementCollection();
        this.listenTo(this.collection, 'add', this._getUser);
        this.listenTo(this.collection, 'all', this.render);
    },

    render() {
        this.$el.html(
            annotationPopover({
                annotations: _.uniq(
                    this.collection.pluck('annotation'),
                    _.property('id')),
                formatDate: this._formatDate,
                users: this._users
            })
        );
        this._show();
        if (!this._visible()) {
            this._hide();
        }
        this._height = this.$('.h-annotation-popover').height();
    },

    _getUser(model) {
        var id = model.get('annotation').get('creatorId');
        if (!_.has(this._users, id)) {
            restRequest({
                path: 'user/' + id
            }).then((user) => {
                this._users[id] = user;
                this.render();
            });
        }
    },

    _formatDate(s) {
        var d = new Date(s);
        return d.toLocaleString();
    },

    _show() {
        if (this._visible()) {
            this.$el.removeClass('hidden');
        }
    },

    _hide() {
        this.$el.addClass('hidden');
    },

    _visible() {
        return !this._hidden && this.collection.length > 0;
    },

    position(evt) {
        if (this._visible()) {
            this.$el.css({
                left: evt.pageX + 5,
                top: evt.pageY - this._height / 2
            });
        }
    },

    toggle(show) {
        this._hidden = !show;
        this.render();
    }
});

export default AnnotationPopover;
